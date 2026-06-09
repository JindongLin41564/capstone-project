# Cloud Build CI/CD option

Cloud Build can replace the GitHub Actions workflow for this project.

With Cloud Build, you do not need these GitHub secrets:

- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_SERVICE_ACCOUNT`

Instead, you connect the GitHub repository to Cloud Build and create a trigger.
Cloud Build runs `cloudbuild.yaml` after a push to `main`.

## File to add

Add this file to the root of your GitHub repository:

- `cloudbuild.yaml`

Keep these files from the previous CI/CD package:

- `requirements-ci.txt`
- `scripts/compile_pipeline.py`
- `scripts/submit_pipeline.py`

## One-time setup

Run this in Cloud Shell:

```bash
export PROJECT_ID="qwiklabs-asl-03-7c1aaee9a503"
export REGION="us-central1"

gcloud config set project "$PROJECT_ID"

gcloud services enable \
  cloudbuild.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com
```

Create a dedicated service account for Cloud Build:

```bash
gcloud iam service-accounts create cloud-build-vertex-runner \
  --display-name="Cloud Build Vertex AI runner"

export BUILD_SA="cloud-build-vertex-runner@$PROJECT_ID.iam.gserviceaccount.com"
```

Grant it the roles needed by this project:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$BUILD_SA" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$BUILD_SA" \
  --role="roles/storage.objectAdmin"

gcloud iam service-accounts add-iam-policy-binding "$BUILD_SA" \
  --member="serviceAccount:$BUILD_SA" \
  --role="roles/iam.serviceAccountUser"
```

## Create the trigger in Console

1. Go to Google Cloud Console.
2. Open `Cloud Build -> Triggers`.
3. Click `Connect repository`.
4. Choose GitHub and authorize the Google Cloud Build GitHub App.
5. Select your repository.
6. Create a trigger:
   - Event: push to branch
   - Branch: `^main$`
   - Configuration: Cloud Build configuration file
   - Location: `/cloudbuild.yaml`
   - Service account: `cloud-build-vertex-runner@qwiklabs-asl-03-7c1aaee9a503.iam.gserviceaccount.com`

After that, every push or merge to `main` starts Cloud Build, uploads the
trainer package to Cloud Storage, compiles the KFP pipeline, and submits a
Vertex AI Pipeline job.
