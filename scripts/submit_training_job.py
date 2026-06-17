"""Submit the GBDT trainer as a Vertex AI Custom Job.

The script keeps the notebook workflow reproducible: build the source package,
upload it to GCS, create or reuse TensorBoard, then submit a CustomJob.
"""

import argparse
import glob
import os
import subprocess
from datetime import datetime
from pathlib import Path

from google.cloud import aiplatform


def load_env_file(path):
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y"}


def required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def gcs_bucket_uri(bucket):
    return bucket if bucket.startswith("gs://") else f"gs://{bucket}"


def default_data_path(bucket_uri, split):
    data_base_uri = os.getenv("DATA_GCS_BASE_URI")
    table_prefix = os.getenv("DATA_TABLE_PREFIX", "project_completion")
    filename_prefix = {"train": "project-train", "valid": "project-valid"}[split]
    if data_base_uri:
        return f"{data_base_uri.rstrip('/')}/{table_prefix}/{split}/{filename_prefix}-*.csv"
    return f"{bucket_uri.rstrip('/')}/jindong_lin/data/{table_prefix}/{split}/{filename_prefix}-*.csv"


def get_compute_service_account(project_id):
    configured = os.getenv("SERVICE_ACCOUNT") or os.getenv("VERTEX_SERVICE_ACCOUNT")
    if configured:
        return configured
    result = subprocess.run(
        [
            "gcloud",
            "iam",
            "service-accounts",
            "list",
            "--project",
            project_id,
            "--filter=email ~ -compute@developer.gserviceaccount.com",
            "--format=value(email)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    accounts = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not accounts:
        raise RuntimeError("No compute service account found. Set SERVICE_ACCOUNT explicitly.")
    return accounts[0]


def create_or_get_tensorboard(project_id, region, bucket_uri, display_name):
    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=f"{bucket_uri.rstrip('/')}/project_completion/staging",
    )
    existing = aiplatform.Tensorboard.list(
        filter=f'display_name="{display_name}"',
        order_by="create_time desc",
        project=project_id,
        location=region,
    )
    if existing:
        return existing[0]
    return aiplatform.Tensorboard.create(
        display_name=display_name,
        project=project_id,
        location=region,
    )


def build_and_upload_package(package_uri):
    subprocess.run(["python", "setup.py", "sdist", "--formats=gztar"], cwd="project_completion", check=True)
    local_package = sorted(glob.glob("project_completion/dist/*.tar.gz"))[-1]
    subprocess.run(["gcloud", "storage", "cp", local_package, package_uri], check=True)
    return local_package


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_file", default="env.txt")
    parser.add_argument("--project_id", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--train_data_path", default=None)
    parser.add_argument("--eval_data_path", default=None)
    parser.add_argument("--schema_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--package_uri", default=None)
    parser.add_argument("--training_image_uri", default=None)
    parser.add_argument("--machine_type", default=None)
    parser.add_argument("--tensorboard_display_name", default=None)
    parser.add_argument("--n_estimators", type=int, default=None)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--random_state", type=int, default=None)
    parser.add_argument("--label_scale", type=float, default=None)
    parser.add_argument("--no_submit", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    load_env_file(args.env_file)

    project_id = args.project_id or required_env("PROJECT_ID")
    region = args.region or os.getenv("REGION", "us-central1")
    bucket_uri = gcs_bucket_uri(args.bucket or required_env("BUCKET"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_data_path = args.train_data_path or os.getenv("TRAIN_DATA_PATH") or default_data_path(bucket_uri, "train")
    eval_data_path = args.eval_data_path or os.getenv("EVAL_DATA_PATH") or default_data_path(bucket_uri, "valid")
    schema_path = args.schema_path or required_env("SCHEMA_PATH")
    output_dir = args.output_dir or f"{bucket_uri}/project_completion/trained_gbdt_model_{timestamp}"
    package_uri = (
        args.package_uri
        or os.getenv("PACKAGE_URI")
        or f"{bucket_uri}/project_completion/packages/project_completion_trainer_{timestamp}.tar.gz"
    )
    training_image_uri = (
        args.training_image_uri
        or os.getenv("TRAINING_IMAGE_URI")
        or "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest"
    )
    machine_type = args.machine_type or os.getenv("VERTEX_MACHINE_TYPE", "n1-standard-4")
    tensorboard_display_name = (
        args.tensorboard_display_name
        or os.getenv("TENSORBOARD_DISPLAY_NAME")
        or "project-forecast-tensorboard"
    )

    n_estimators = args.n_estimators or int(os.getenv("N_ESTIMATORS", "500"))
    max_depth = args.max_depth or int(os.getenv("MAX_DEPTH", "8"))
    learning_rate = args.learning_rate or float(os.getenv("GBDT_LEARNING_RATE", "0.1"))
    random_state = args.random_state or int(os.getenv("RANDOM_STATE", "42"))
    label_scale = args.label_scale or float(os.getenv("LABEL_SCALE", "1.0"))

    tensorboard = create_or_get_tensorboard(project_id, region, bucket_uri, tensorboard_display_name)
    service_account = get_compute_service_account(project_id)
    local_package = build_and_upload_package(package_uri)

    train_args = [
        "--train_data_path", train_data_path,
        "--eval_data_path", eval_data_path,
        "--schema_path", schema_path,
        "--output_dir", output_dir,
        "--n_estimators", str(n_estimators),
        "--max_depth", str(max_depth),
        "--learning_rate", str(learning_rate),
        "--random_state", str(random_state),
        "--label_scale", str(label_scale),
    ]

    worker_pool_specs = [
        {
            "machine_spec": {"machine_type": machine_type},
            "replica_count": 1,
            "python_package_spec": {
                "executor_image_uri": training_image_uri,
                "package_uris": [package_uri],
                "python_module": "trainer.task",
                "args": train_args,
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name=f"project_completion_gbdt_training_{timestamp}",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=output_dir,
    )

    should_submit = env_bool("RUN_VERTEX_CUSTOM_JOB", True) and not args.no_submit
    print("Local package:", local_package)
    print("PACKAGE_URI:", package_uri)
    print("MODEL_OUTPUT_DIR:", output_dir)
    print("TensorBoard:", tensorboard.resource_name)
    print("Service account:", service_account)
    print("Training args:", train_args)

    if should_submit:
        job.submit(service_account=service_account, tensorboard=tensorboard.resource_name)
        print("Submitted Custom Job:", job.resource_name)
    else:
        print("Custom Job is configured but not submitted.")


if __name__ == "__main__":
    main()
