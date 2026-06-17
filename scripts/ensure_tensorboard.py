"""Create or reuse the Vertex AI TensorBoard used by custom training jobs."""

import contextlib
import os
import sys

from google.cloud import aiplatform


def required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    project_id = required_env("PROJECT_ID")
    bucket = required_env("BUCKET")
    region = os.getenv("REGION", "us-central1")
    display_name = os.getenv("TENSORBOARD_DISPLAY_NAME", "project-forecast-tensorboard")

    with contextlib.redirect_stdout(sys.stderr):
        aiplatform.init(
            project=project_id,
            location=region,
            staging_bucket=f"gs://{bucket}/project_completion/staging",
        )
        existing_tensorboards = aiplatform.Tensorboard.list(
            filter=f'display_name="{display_name}"',
            order_by="create_time desc",
        )
        if existing_tensorboards:
            tensorboard = existing_tensorboards[0]
        else:
            tensorboard = aiplatform.Tensorboard.create(
                display_name=display_name,
                project=project_id,
                location=region,
            )

    print(tensorboard.resource_name)
