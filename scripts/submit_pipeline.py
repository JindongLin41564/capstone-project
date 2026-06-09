"""Submit the compiled pipeline YAML to Vertex AI Pipelines."""

import os
from datetime import datetime, timezone

from google.cloud import aiplatform


def required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    project_id = required_env("PROJECT_ID")
    region = os.getenv("REGION", "us-central1")
    pipeline_root = required_env("PIPELINE_ROOT")
    pipeline_name = os.getenv("PIPELINE_NAME", "project-completion-dnn")
    service_account = os.getenv("VERTEX_SERVICE_ACCOUNT") or os.getenv("SERVICE_ACCOUNT")

    aiplatform.init(project=project_id, location=region, staging_bucket=pipeline_root)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job = aiplatform.PipelineJob(
        display_name=f"{pipeline_name}-{timestamp}",
        template_path="project_completion_kfp_pipeline.yaml",
        pipeline_root=pipeline_root,
        enable_caching=False,
    )
    job.submit(service_account=service_account)
    print(f"Submitted Vertex AI Pipeline job: {job.resource_name}")
