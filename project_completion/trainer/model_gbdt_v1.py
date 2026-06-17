"""Train and evaluate a GBDT model for project completion time prediction."""

import json
import os
import tempfile
from pathlib import Path

# TensorFlow is only used for GCS-compatible file IO and TensorBoard summaries.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def infer_schema_path(train_data_path):
    if "/train/" in train_data_path:
        return train_data_path.split("/train/", 1)[0] + "/schema/feature_schema.json"
    return ""


def load_feature_schema(schema_path):
    with tf.io.gfile.GFile(schema_path, "r") as f:
        schema = json.load(f)

    required_keys = ["label", "categorical_features", "numeric_features", "csv_columns"]
    missing = [key for key in required_keys if key not in schema]
    if missing:
        raise ValueError(f"Feature schema is missing required keys: {missing}")
    if not schema["categorical_features"] and not schema["numeric_features"]:
        raise ValueError("Feature schema must define at least one categorical or numeric feature")
    return schema


def expand_files(pattern):
    files = tf.io.gfile.glob(pattern)
    if not files:
        raise ValueError(f"No CSV files matched: {pattern}")
    return sorted(files)


def read_csv_dataset(pattern, schema):
    frames = []
    for path in expand_files(pattern):
        with tf.io.gfile.GFile(path, "r") as f:
            frames.append(
                pd.read_csv(
                    f,
                    header=None,
                    names=schema["csv_columns"],
                    na_values=["", "NULL", "null", "NaN"],
                    keep_default_na=True,
                )
            )
    if not frames:
        raise ValueError(f"No rows loaded from: {pattern}")
    data = pd.concat(frames, ignore_index=True)
    return data


def split_features_label(data, schema, label_scale=1.0):
    label = schema["label"]
    feature_names = schema["categorical_features"] + schema["numeric_features"]
    missing = [name for name in [label] + feature_names if name not in data.columns]
    if missing:
        raise ValueError(f"Input data is missing columns required by schema: {missing}")

    x = data[feature_names].copy()
    for name in schema["categorical_features"]:
        x[name] = x[name].fillna("UNKNOWN").astype(str)
    for name in schema["numeric_features"]:
        x[name] = pd.to_numeric(x[name], errors="coerce")

    y = pd.to_numeric(data[label], errors="coerce") / float(label_scale)
    valid = y.notna()
    if not valid.all():
        x = x.loc[valid].reset_index(drop=True)
        y = y.loc[valid].reset_index(drop=True)
    return x, y.to_numpy(dtype=np.float64)


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_gbdt_model(schema, n_estimators, max_depth, learning_rate, random_state):
    transformers = []
    if schema["categorical_features"]:
        transformers.append(("categorical", make_one_hot_encoder(), schema["categorical_features"]))
    if schema["numeric_features"]:
        numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        transformers.append(("numeric", numeric_pipeline, schema["numeric_features"]))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    regressor = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocessor), ("model", regressor)])


def regression_metrics(y_true, y_pred, label_scale):
    true_days = np.asarray(y_true) * label_scale
    pred_days = np.asarray(y_pred) * label_scale
    mse = mean_squared_error(true_days, pred_days)
    return {
        "eval_loss_mse": float(mse),
        "eval_mae_days": float(mean_absolute_error(true_days, pred_days)),
        "eval_rmse_days": float(np.sqrt(mse)),
        "eval_rows": int(true_days.shape[0]),
    }


def write_json(path, payload):
    with tf.io.gfile.GFile(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def save_joblib_gcs_compatible(model, output_path):
    if output_path.startswith("gs://"):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "model.joblib")
            joblib.dump(model, local_path)
            tf.io.gfile.copy(local_path, output_path, overwrite=True)
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)


def write_tensorboard_logs(model, x_train, y_train, x_eval, y_eval, tensorboard_dir, label_scale):
    writer = tf.summary.create_file_writer(tensorboard_dir)
    regressor = model.named_steps["model"]
    preprocessor = model.named_steps["preprocess"]
    train_encoded = preprocessor.transform(x_train)
    eval_encoded = preprocessor.transform(x_eval)

    with writer.as_default():
        for step, (train_pred, eval_pred) in enumerate(
            zip(regressor.staged_predict(train_encoded), regressor.staged_predict(eval_encoded)), start=1
        ):
            train_true_days = y_train * label_scale
            train_pred_days = train_pred * label_scale
            eval_true_days = y_eval * label_scale
            eval_pred_days = eval_pred * label_scale
            tf.summary.scalar("train_mae_days", mean_absolute_error(train_true_days, train_pred_days), step=step)
            tf.summary.scalar("eval_mae_days", mean_absolute_error(eval_true_days, eval_pred_days), step=step)
            tf.summary.scalar(
                "eval_rmse_days",
                np.sqrt(mean_squared_error(eval_true_days, eval_pred_days)),
                step=step,
            )
        writer.flush()


def train_and_evaluate(hparams):
    n_estimators = int(hparams.get("n_estimators", 500))
    max_depth = int(hparams.get("max_depth", 8))
    learning_rate = float(hparams.get("learning_rate", 0.1))
    random_state = int(hparams.get("random_state", 42))
    label_scale = float(hparams.get("label_scale", 1.0))

    output_dir = hparams["output_dir"].rstrip("/")
    train_data_path = hparams["train_data_path"]
    eval_data_path = hparams["eval_data_path"]
    schema_path = hparams.get("schema_path") or os.environ.get("SCHEMA_PATH") or infer_schema_path(train_data_path)
    if not schema_path:
        raise ValueError("schema_path is required, or train_data_path must contain '/train/'")

    schema = load_feature_schema(schema_path)
    if tf.io.gfile.exists(output_dir):
        tf.io.gfile.rmtree(output_dir)
    tf.io.gfile.makedirs(output_dir)

    tensorboard_dir = os.environ.get("AIP_TENSORBOARD_LOG_DIR") or os.path.join(output_dir, "tensorboard")
    model_path = os.path.join(output_dir, "model.joblib")
    metrics_path = os.path.join(output_dir, "metrics.json")

    train_df = read_csv_dataset(train_data_path, schema)
    eval_df = read_csv_dataset(eval_data_path, schema)
    x_train, y_train = split_features_label(train_df, schema, label_scale=label_scale)
    x_eval, y_eval = split_features_label(eval_df, schema, label_scale=label_scale)

    model = build_gbdt_model(
        schema=schema,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(x_train, y_train)

    eval_pred = model.predict(x_eval)
    metrics = regression_metrics(y_eval, eval_pred, label_scale=label_scale)
    train_pred = model.predict(x_train)
    train_mse = mean_squared_error(y_train * label_scale, train_pred * label_scale)
    metrics.update(
        {
            "train_mae_days": float(mean_absolute_error(y_train * label_scale, train_pred * label_scale)),
            "train_rmse_days": float(np.sqrt(train_mse)),
            "train_rows": int(y_train.shape[0]),
            "model_type": "sklearn.ensemble.GradientBoostingRegressor",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
            "label_scale": label_scale,
            "schema_path": schema_path,
            "feature_schema": schema,
            "tensorboard_dir": tensorboard_dir,
            "model_path": model_path,
        }
    )

    write_tensorboard_logs(model, x_train, y_train, x_eval, y_eval, tensorboard_dir, label_scale)
    save_joblib_gcs_compatible(model, model_path)
    write_json(metrics_path, metrics)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved GBDT model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"TensorBoard logs: {tensorboard_dir}")
