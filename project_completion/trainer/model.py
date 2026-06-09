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
LABEL = "days_to_S90"
CATEGORICAL_FEATURES = ["RID", "ZDP", "GID", "MID"]
NUMERIC_FEATURES = [
    "days_S44_from_S30",
    "days_S51_from_S30",
    "days_S52_from_S30",
    "days_S56_from_S30",
    "days_S68_from_S30",
    "days_S71_from_S30",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
COLUMNS = [
    LABEL,
    "RID", "ZDP", "GID", "MID",
    "days_S44_from_S30", "days_S51_from_S30", "days_S52_from_S30",
    "days_S56_from_S30", "days_S68_from_S30", "days_S71_from_S30",
    "SID", "PID", "key",
]
CSV_DEFAULTS = [
    [0.0],
    ["UNKNOWN"], ["UNKNOWN"], ["UNKNOWN"], ["UNKNOWN"],
    [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
    [""], [""], ["unused"],
]


def parse_csv(row, label_scale=1.0):
    fields = tf.io.decode_csv(row, record_defaults=CSV_DEFAULTS)
    values = dict(zip(COLUMNS, fields))
    label = values.pop(LABEL)
    label = tf.cast(label, tf.float32) / tf.cast(label_scale, tf.float32)

    features = {}
    for name in CATEGORICAL_FEATURES:
        features[name] = tf.reshape(values[name], [1])
    for name in NUMERIC_FEATURES:
        features[name] = tf.reshape(values[name], [1])

    return features, label


def create_dataset(pattern, batch_size, num_repeat=1, mode="eval", label_scale=1.0):
    ds = tf.data.Dataset.list_files(pattern, shuffle=(mode == "train"))
    ds = ds.interleave(
        tf.data.TextLineDataset,
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.map(
        lambda row: parse_csv(row, label_scale=label_scale),
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


def build_preprocessing_layers(train_data_path, batch_size):
    sample_ds = create_dataset(
        train_data_path,
        batch_size=batch_size,
        num_repeat=1,
        mode="eval",
        label_scale=1.0,
    )

    lookups = {}
    for name in CATEGORICAL_FEATURES:
        lookup = StringLookup(
            output_mode="int",
            num_oov_indices=1,
            mask_token=None,
            name=f"{name}_lookup",
        )
        lookup.adapt(sample_ds.map(lambda features, label, n=name: features[n]))
        lookups[name] = lookup

    normalizer = Normalization(axis=-1, name="numeric_normalization")
    numeric_ds = sample_ds.map(
        lambda features, label: tf.concat(
            [tf.cast(features[name], tf.float32) for name in NUMERIC_FEATURES], axis=-1
        )
    )
    normalizer.adapt(numeric_ds)

    return lookups, normalizer


def embedding_dim(vocabulary_size):
    return min(16, max(2, int(np.ceil(np.sqrt(vocabulary_size)))))


def build_dnn_model(hidden_units, learning_rate, dropout_rate, lookups, normalizer):
    inputs = {
        name: Input(name=name, shape=(1,), dtype="string")
        for name in CATEGORICAL_FEATURES
    }
    inputs.update(
        {
            name: Input(name=name, shape=(1,), dtype="float32")
            for name in NUMERIC_FEATURES
        }
    )

    encoded_features = []
    embedding_config = {}
    for name in CATEGORICAL_FEATURES:
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

    numeric_values = Concatenate(name="numeric_features")(
        [inputs[name] for name in NUMERIC_FEATURES]
    )
    encoded_features.append(normalizer(numeric_values))

    x = Concatenate(name="all_features")(encoded_features)
    for index, units in enumerate(hidden_units):
        x = Dense(units, activation="relu", name=f"hidden_{index + 1}")(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name=f"dropout_{index + 1}")(x)

    # The model predicts scaled days_to_S90. Metrics during fit are scaled too;
    # final metrics below are converted back to original day units.
    output = Dense(1, name="scaled_days_to_S90")(x)
    model = keras.Model(inputs=list(inputs.values()), outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
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

    lookups, normalizer = build_preprocessing_layers(train_data_path, batch_size)
    model, embedding_config = build_dnn_model(
        hidden_units, learning_rate, dropout_rate, lookups, normalizer
    )
    model.summary(print_fn=logging.info)
    print("Embedding config:", json.dumps(embedding_config, indent=2))

    train_ds = create_dataset(
        train_data_path,
        batch_size=batch_size,
        num_repeat=None if train_steps_per_epoch else 1,
        mode="train",
        label_scale=label_scale,
    )
    eval_ds = create_dataset(
        eval_data_path,
        batch_size=batch_size,
        num_repeat=None if validation_steps else 1,
        mode="eval",
        label_scale=label_scale,
    )
    eval_ds_for_metrics = create_dataset(
        eval_data_path,
        batch_size=batch_size,
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
