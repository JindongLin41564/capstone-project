"""Train and evaluate an embedding DNN for project completion time prediction."""

import json
import logging
import os

# The local notebook smoke test uses CPU. This avoids noisy CUDA/cuDNN/cuBLAS
# registration messages in Workbench images that do not expose a working GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import tensorflow as tf
from keras import callbacks
from keras.layers import Concatenate, Dense, Dropout, Embedding, Flatten, Input, Normalization, StringLookup
from keras.metrics import MeanAbsoluteError, RootMeanSquaredError

AUTOTUNE = tf.data.AUTOTUNE


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


def csv_defaults(schema):
    label = schema["label"]
    categorical_features = set(schema["categorical_features"])
    numeric_features = set(schema["numeric_features"])
    defaults = []

    for column in schema["csv_columns"]:
        if column == label or column in numeric_features:
            defaults.append([0.0])
        elif column in categorical_features:
            defaults.append(["UNKNOWN"])
        else:
            defaults.append([""])
    return defaults


def parse_csv(row, schema, label_scale=1.0):
    fields = tf.io.decode_csv(row, record_defaults=csv_defaults(schema))
    values = dict(zip(schema["csv_columns"], fields))
    label = values.pop(schema["label"])
    label = tf.cast(label, tf.float32) / tf.cast(label_scale, tf.float32)

    features = {}
    for name in schema["categorical_features"]:
        features[name] = tf.reshape(values[name], [1])
    for name in schema["numeric_features"]:
        features[name] = tf.reshape(values[name], [1])

    return features, label


def create_dataset(pattern, batch_size, schema, num_repeat=1, mode="eval", label_scale=1.0):
    ds = tf.data.Dataset.list_files(pattern, shuffle=(mode == "train"))
    ds = ds.interleave(
        tf.data.TextLineDataset,
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.map(
        lambda row: parse_csv(row, schema=schema, label_scale=label_scale),
        num_parallel_calls=AUTOTUNE,
    )
    if mode == "train":
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    if num_repeat is None:
        ds = ds.repeat()
    elif num_repeat != 1:
        ds = ds.repeat(num_repeat)
    return ds


def build_preprocessing_layers(train_data_path, batch_size, schema):
    sample_ds = create_dataset(
        train_data_path,
        batch_size=batch_size,
        schema=schema,
        num_repeat=1,
        mode="eval",
        label_scale=1.0,
    )

    lookups = {}
    for name in schema["categorical_features"]:
        lookup = StringLookup(
            output_mode="int",
            num_oov_indices=1,
            mask_token=None,
            name=f"{name}_lookup",
        )
        lookup.adapt(sample_ds.map(lambda features, label, n=name: features[n]))
        lookups[name] = lookup

    normalizer = None
    if schema["numeric_features"]:
        normalizer = Normalization(axis=-1, name="numeric_normalization")
        numeric_ds = sample_ds.map(
            lambda features, label: tf.concat(
                [tf.cast(features[name], tf.float32) for name in schema["numeric_features"]], axis=-1
            )
        )
        normalizer.adapt(numeric_ds)

    return lookups, normalizer


def embedding_dim(vocabulary_size):
    return min(16, max(2, int(np.ceil(np.sqrt(vocabulary_size)))))


def build_dnn_model(hidden_units, learning_rate, dropout_rate, lookups, normalizer, schema):
    inputs = {
        name: Input(name=name, shape=(1,), dtype="string")
        for name in schema["categorical_features"]
    }
    inputs.update(
        {
            name: Input(name=name, shape=(1,), dtype="float32")
            for name in schema["numeric_features"]
        }
    )

    encoded_features = []
    embedding_config = {}
    for name in schema["categorical_features"]:
        lookup = lookups[name]
        vocab_size = lookup.vocabulary_size()
        dim = embedding_dim(vocab_size)
        embedding_config[name] = {"vocabulary_size": int(vocab_size), "embedding_dim": int(dim)}
        category_id = lookup(inputs[name])
        category_embedding = Embedding(
            input_dim=vocab_size,
            output_dim=dim,
            name=f"{name}_embedding",
        )(category_id)
        encoded_features.append(Flatten(name=f"{name}_embedding_flatten")(category_embedding))

    if schema["numeric_features"]:
        if len(schema["numeric_features"]) == 1:
            numeric_values = inputs[schema["numeric_features"][0]]
        else:
            numeric_values = Concatenate(name="numeric_features")(
                [inputs[name] for name in schema["numeric_features"]]
            )
        encoded_features.append(normalizer(numeric_values))

    if len(encoded_features) == 1:
        x = encoded_features[0]
    else:
        x = Concatenate(name="all_features")(encoded_features)
    for index, units in enumerate(hidden_units):
        x = Dense(units, activation="relu", name=f"hidden_{index + 1}")(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name=f"dropout_{index + 1}")(x)

    # The model predicts scaled days_to_S90. Metrics during fit are scaled too;
    # final metrics below are converted back to original day units.
    output = Dense(1, name="scaled_days_to_S90")(x)
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.Huber(delta=0.25),
        metrics=[MeanAbsoluteError(name="scaled_mae"), RootMeanSquaredError(name="scaled_rmse")],
    )
    return model, embedding_config


def write_json_gcs(path, payload):
    with tf.io.gfile.GFile(path, "w") as f:
        json.dump(payload, f, indent=2)


def evaluate_in_days(model, dataset, label_scale):
    y_true_batches = []
    y_pred_batches = []
    for features, scaled_label in dataset:
        scaled_pred = model.predict(features, verbose=0)
        y_true_batches.append(tf.reshape(scaled_label, [-1]).numpy() * label_scale)
        y_pred_batches.append(tf.reshape(scaled_pred, [-1]).numpy() * label_scale)

    y_true = np.concatenate(y_true_batches)
    y_pred = np.concatenate(y_pred_batches)
    error = y_pred - y_true
    return {
        "eval_loss_mse": float(np.mean(np.square(error))),
        "eval_mae_days": float(np.mean(np.abs(error))),
        "eval_rmse_days": float(np.sqrt(np.mean(np.square(error)))),
        "eval_rows": int(y_true.shape[0]),
    }


def train_and_evaluate(hparams):
    batch_size = hparams["batch_size"]
    epochs = hparams["epochs"]
    learning_rate = hparams["learning_rate"]
    dropout_rate = hparams["dropout_rate"]
    label_scale = hparams["label_scale"]
    hidden_units = [int(value) for value in hparams["hidden_units"].split()]
    train_steps_per_epoch = hparams.get("train_steps_per_epoch")
    validation_steps = hparams.get("validation_steps")
    num_examples_to_train_on = hparams.get("num_examples_to_train_on")
    if num_examples_to_train_on is None and train_steps_per_epoch:
        num_examples_to_train_on = train_steps_per_epoch * batch_size
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

    tensorboard_dir = os.environ.get("AIP_TENSORBOARD_LOG_DIR")
    if not tensorboard_dir:
        tensorboard_dir = os.path.join(output_dir, "tensorboard")

    checkpoint_path = os.path.join(output_dir, "checkpoints", "best_model.keras")
    model_export_path = os.path.join(output_dir, "model.keras")
    serving_model_export_path = os.path.join(output_dir, "savedmodel")
    metrics_path = os.path.join(output_dir, "metrics.json")

    lookups, normalizer = build_preprocessing_layers(train_data_path, batch_size, schema)
    model, embedding_config = build_dnn_model(
        hidden_units, learning_rate, dropout_rate, lookups, normalizer, schema
    )
    model.summary(print_fn=logging.info)
    print("Feature schema:", json.dumps(schema, indent=2))
    print("Embedding config:", json.dumps(embedding_config, indent=2))

    train_ds = create_dataset(
        train_data_path,
        batch_size=batch_size,
        schema=schema,
        num_repeat=None if train_steps_per_epoch else 1,
        mode="train",
        label_scale=label_scale,
    )
    eval_ds = create_dataset(
        eval_data_path,
        batch_size=batch_size,
        schema=schema,
        num_repeat=None if validation_steps else 1,
        mode="eval",
        label_scale=label_scale,
    )
    eval_ds_for_metrics = create_dataset(
        eval_data_path,
        batch_size=batch_size,
        schema=schema,
        num_repeat=1,
        mode="eval",
        label_scale=label_scale,
    )
    history = model.fit(
        train_ds,
        validation_data=eval_ds,
        epochs=epochs,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
        callbacks=[
            callbacks.ModelCheckpoint(checkpoint_path, monitor="val_scaled_mae", save_best_only=True),
            callbacks.EarlyStopping(monitor="val_scaled_mae", patience=5, restore_best_weights=True),
            callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=0),
        ],
    )

    metrics = evaluate_in_days(model, eval_ds_for_metrics, label_scale)
    metrics.update(
        {
            "epochs_ran": len(history.history.get("loss", [])),
            "num_examples_to_train_on_config": num_examples_to_train_on,
            "hidden_units": hidden_units,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "label_scale": label_scale,
            "schema_path": schema_path,
            "feature_schema": schema,
            "embedding_config": embedding_config,
            "tensorboard_dir": tensorboard_dir,
        }
    )
    print(json.dumps(metrics, indent=2))

    model.save(model_export_path)
    model.export(serving_model_export_path)
    write_json_gcs(metrics_path, metrics)

    print(f"Saved Keras model to: {model_export_path}")
    print(f"Saved serving model to: {serving_model_export_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"TensorBoard logs: {tensorboard_dir}")
