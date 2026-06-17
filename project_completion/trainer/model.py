"""Train and evaluate a GBDT model for project completion time prediction."""

import json
import os
import platform
import tempfile
from pathlib import Path

# TensorFlow is only used for GCS-compatible file IO and TensorBoard summaries.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
        "eval_r2": float(r2_score(true_days, pred_days)),
        "eval_rows": int(true_days.shape[0]),
    }


def write_json(path, payload):
    with tf.io.gfile.GFile(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path):
    with tf.io.gfile.GFile(path, "r") as f:
        return json.load(f)


def write_dataframe_csv(path, data):
    with tf.io.gfile.GFile(path, "w") as f:
        data.to_csv(f, index=False)


def infer_original_feature_name(encoded_feature_name, schema):
    name = encoded_feature_name.split("__", 1)[-1]
    for feature_name in sorted(schema["categorical_features"], key=len, reverse=True):
        if name == feature_name or name.startswith(f"{feature_name}_"):
            return feature_name
    for feature_name in schema["numeric_features"]:
        if name == feature_name:
            return feature_name
    return name


def build_feature_importance_frames(model, schema):
    preprocessor = model.named_steps["preprocess"]
    regressor = model.named_steps["model"]
    encoded_feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_

    encoded_importance = (
        pd.DataFrame(
            {
                "encoded_feature": encoded_feature_names,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    encoded_importance["original_feature"] = encoded_importance["encoded_feature"].map(
        lambda name: infer_original_feature_name(name, schema)
    )

    original_importance = (
        encoded_importance.groupby("original_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return encoded_importance, original_importance


def save_joblib_gcs_compatible(model, output_path):
    if output_path.startswith("gs://"):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "model.joblib")
            joblib.dump(model, local_path)
            tf.io.gfile.copy(local_path, output_path, overwrite=True)
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)


def model_dir_from_path(model_path):
    return model_path.rsplit("/", 1)[0]


def dependency_metadata():
    return {
        "python_version": platform.python_version(),
        "joblib_version": joblib.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "tensorflow_version": tf.__version__,
    }


def assert_model_dependency_compatible(model_path):
    metadata_path = os.path.join(model_dir_from_path(model_path), "model_metadata.json")
    if not tf.io.gfile.exists(metadata_path):
        raise RuntimeError(
            "model_metadata.json was not found next to model.joblib. "
            "This is probably an old model artifact. Retrain to a new output_dir "
            "with the current project code before loading."
        )

    metadata = read_json(metadata_path)
    trained_versions = metadata.get("dependency_versions", {})
    current_versions = dependency_metadata()
    checks = [
        ("scikit_learn_version", "scikit-learn"),
        ("joblib_version", "joblib"),
    ]
    mismatches = []
    for key, label in checks:
        trained = trained_versions.get(key)
        current = current_versions.get(key)
        if trained and current and trained != current:
            mismatches.append(f"{label}: trained={trained}, current={current}")

    if mismatches:
        raise RuntimeError(
            "Model dependency version mismatch. Do not load this model.joblib. "
            "Retrain with the current environment or use the exact training environment. "
            + "; ".join(mismatches)
        )
    return metadata


def load_joblib_gcs_compatible(model_path):
    assert_model_dependency_compatible(model_path)
    if model_path.startswith("gs://"):
        with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
            tf.io.gfile.copy(model_path, tmp.name, overwrite=True)
            return joblib.load(tmp.name)
    return joblib.load(model_path)


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
            tf.summary.scalar("eval_r2", r2_score(eval_true_days, eval_pred_days), step=step)
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
    metadata_path = os.path.join(output_dir, "model_metadata.json")

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
    y_train_days = y_train * label_scale
    train_pred_days = train_pred * label_scale
    train_mse = mean_squared_error(y_train_days, train_pred_days)

    encoded_importance, original_importance = build_feature_importance_frames(model, schema)
    encoded_importance_path = os.path.join(output_dir, "feature_importance_encoded.csv")
    original_importance_path = os.path.join(output_dir, "feature_importance_by_original_feature.csv")
    metrics.update(
        {
            "train_mae_days": float(mean_absolute_error(y_train_days, train_pred_days)),
            "train_rmse_days": float(np.sqrt(train_mse)),
            "train_r2": float(r2_score(y_train_days, train_pred_days)),
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
            "feature_importance_encoded_path": encoded_importance_path,
            "feature_importance_by_original_feature_path": original_importance_path,
            "dependency_versions": dependency_metadata(),
        }
    )

    metadata = {
        "model_path": model_path,
        "model_type": "sklearn.ensemble.GradientBoostingRegressor",
        "dependency_versions": dependency_metadata(),
        "training_hparams": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
            "label_scale": label_scale,
        },
        "schema_path": schema_path,
    }

    write_tensorboard_logs(model, x_train, y_train, x_eval, y_eval, tensorboard_dir, label_scale)
    save_joblib_gcs_compatible(model, model_path)
    write_dataframe_csv(encoded_importance_path, encoded_importance)
    write_dataframe_csv(original_importance_path, original_importance)
    write_json(metrics_path, metrics)
    write_json(metadata_path, metadata)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("Top feature importance by original feature:")
    print(original_importance.head(20).to_string(index=False))
    print(f"Saved GBDT model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved model metadata to: {metadata_path}")
    print(f"Saved encoded feature importance to: {encoded_importance_path}")
    print(f"Saved original feature importance to: {original_importance_path}")
    print(f"TensorBoard logs: {tensorboard_dir}")
