"""Vertex AI Pipeline for the project completion DNN trainer."""

import math
import os
from datetime import datetime

from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from kfp import dsl


PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")
PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
PACKAGE_URI = os.getenv("PACKAGE_URI")
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH")
EVAL_DATA_PATH = os.getenv("EVAL_DATA_PATH")
SCHEMA_PATH = os.getenv("SCHEMA_PATH")
MODEL_OUTPUT_DIR = os.getenv(
    "MODEL_OUTPUT_DIR",
    f"{PIPELINE_ROOT}/trained_dnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)

TRAINING_IMAGE_URI = os.getenv(
    "TRAINING_IMAGE_URI",
    "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-15.py310:latest",
)
SERVING_IMAGE_URI = os.getenv(
    "SERVING_IMAGE_URI",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest",
)

PIPELINE_NAME = os.getenv("PIPELINE_NAME", "project-completion-dnn")
MODEL_DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME", PIPELINE_NAME)
ENDPOINT_DISPLAY_NAME = os.getenv("ENDPOINT_DISPLAY_NAME", f"{PIPELINE_NAME}-endpoint")
SERVING_MACHINE_TYPE = os.getenv("SERVING_MACHINE_TYPE", "n1-standard-2")
SERVING_MIN_REPLICA_COUNT = int(os.getenv("SERVING_MIN_REPLICA_COUNT", "1"))
SERVING_MAX_REPLICA_COUNT = int(os.getenv("SERVING_MAX_REPLICA_COUNT", "1"))
HIDDEN_UNITS = os.getenv("HIDDEN_UNITS", "128 64 32")
SERVICE_ACCOUNT = os.getenv("VERTEX_SERVICE_ACCOUNT") or os.getenv("SERVICE_ACCOUNT")
TENSORBOARD_RESOURCE_NAME = os.getenv("TENSORBOARD_RESOURCE_NAME") or os.getenv("VERTEX_TENSORBOARD")


def _env_int(name, default):
    return int(os.getenv(name, str(default)))


def _env_float(name, default):
    return float(os.getenv(name, str(default)))


def _resolve_steps(env_name, table_env_name, batch_size):
    raw_value = os.getenv(env_name, "AUTO")
    if raw_value.upper() != "AUTO":
        return int(raw_value)

    table_name = os.getenv(table_env_name)
    if not table_name:
        raise ValueError(f"{env_name}=AUTO requires {table_env_name} to be set")

    from google.cloud import bigquery

    bq_client = bigquery.Client(project=PROJECT_ID)
    rows = bq_client.get_table(table_name).num_rows
    if rows == 0:
        raise ValueError(f"{table_name} has 0 rows")

    steps = max(1, math.ceil(rows / batch_size))
    print(f"{env_name}=AUTO -> {steps} steps from {rows} rows in {table_name}")
    return steps


missing = [
    name for name, value in {
        "PROJECT_ID": PROJECT_ID,
        "PIPELINE_ROOT": PIPELINE_ROOT,
        "PACKAGE_URI": PACKAGE_URI,
        "TRAIN_DATA_PATH": TRAIN_DATA_PATH,
        "EVAL_DATA_PATH": EVAL_DATA_PATH,
        "MODEL_OUTPUT_DIR": MODEL_OUTPUT_DIR,
    }.items() if not value
]
if missing:
    raise ValueError(f"Missing required environment variables for pipeline compilation: {missing}")

BATCH_SIZE = _env_int("BATCH_SIZE", 64)
EPOCHS = _env_int("EPOCHS", 10)
TRAIN_STEPS_PER_EPOCH = _resolve_steps("TRAIN_STEPS_PER_EPOCH", "TRAIN_TABLE", BATCH_SIZE)
VALIDATION_STEPS = _resolve_steps("VALIDATION_STEPS", "VALID_TABLE", BATCH_SIZE)
LEARNING_RATE = _env_float("LEARNING_RATE", 0.001)
DROPOUT_RATE = _env_float("DROPOUT_RATE", 0.0)
LABEL_SCALE = _env_float("LABEL_SCALE", 365.0)


@dsl.pipeline(
    name=f"{PIPELINE_NAME}-pipeline",
    description="Train, register, and deploy the project completion DNN on Vertex AI.",
    pipeline_root=PIPELINE_ROOT,
)
def create_pipeline():
    train_args = [
        "--train_data_path", TRAIN_DATA_PATH,
        "--eval_data_path", EVAL_DATA_PATH,
        "--output_dir", MODEL_OUTPUT_DIR,
        "--batch_size", str(BATCH_SIZE),
        "--epochs", str(EPOCHS),
        "--train_steps_per_epoch", str(TRAIN_STEPS_PER_EPOCH),
        "--validation_steps", str(VALIDATION_STEPS),
        "--hidden_units", HIDDEN_UNITS,
        "--learning_rate", str(LEARNING_RATE),
        "--dropout_rate", str(DROPOUT_RATE),
        "--label_scale", str(LABEL_SCALE),
    ]
    if SCHEMA_PATH:
        train_args.extend(["--schema_path", SCHEMA_PATH])

    worker_pool_specs = [
        {
            "machine_spec": {"machine_type": "n1-standard-4"},
            "replica_count": 1,
            "python_package_spec": {
                "executor_image_uri": TRAINING_IMAGE_URI,
                "package_uris": [PACKAGE_URI],
                "python_module": "trainer.task",
                "args": train_args,
            },
        }
    ]

    custom_training_job_args = {
        "project": PROJECT_ID,
        "location": REGION,
        "display_name": f"{PIPELINE_NAME}-training-job",
        "worker_pool_specs": worker_pool_specs,
        "base_output_directory": MODEL_OUTPUT_DIR,
    }
    if SERVICE_ACCOUNT:
        custom_training_job_args["service_account"] = SERVICE_ACCOUNT
    if TENSORBOARD_RESOURCE_NAME:
        custom_training_job_args["tensorboard"] = TENSORBOARD_RESOURCE_NAME

    training_task = CustomTrainingJobOp(**custom_training_job_args)

    saved_model_importer = dsl.importer(
        artifact_uri=f"{MODEL_OUTPUT_DIR}/savedmodel",
        artifact_class=artifact_types.UnmanagedContainerModel,
        metadata={"containerSpec": {"imageUri": SERVING_IMAGE_URI}},
    )
    saved_model_importer.after(training_task)

    model_upload_task = ModelUploadOp(
        project=PROJECT_ID,
        location=REGION,
        display_name=f"{MODEL_DISPLAY_NAME}-model",
        unmanaged_container_model=saved_model_importer.output,
    )
    model_upload_task.after(saved_model_importer)

    endpoint_create_task = EndpointCreateOp(
        project=PROJECT_ID,
        location=REGION,
        display_name=ENDPOINT_DISPLAY_NAME,
    )
    endpoint_create_task.after(model_upload_task)

    model_deploy_task = ModelDeployOp(
        model=model_upload_task.outputs["model"],
        endpoint=endpoint_create_task.outputs["endpoint"],
        deployed_model_display_name=MODEL_DISPLAY_NAME,
        dedicated_resources_machine_type=SERVING_MACHINE_TYPE,
        dedicated_resources_min_replica_count=SERVING_MIN_REPLICA_COUNT,
        dedicated_resources_max_replica_count=SERVING_MAX_REPLICA_COUNT,
    )
    model_deploy_task.after(endpoint_create_task)
