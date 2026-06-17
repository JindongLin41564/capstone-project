# Project Completion GBDT

Minimal local and Vertex AI Custom Training project for project completion time prediction.

## Install

```bash
pip install -r requirements-local.txt
```

## Local Training

```bash
cd project_completion
python -m trainer.task \
  --train_data_path "gs://.../project_completion/train/project-train-*.csv" \
  --eval_data_path "gs://.../project_completion/valid/project-valid-*.csv" \
  --schema_path "gs://.../project_completion/schema/feature_schema.json" \
  --output_dir "gs://.../project_completion/local_gbdt_test" \
  --n_estimators 500 \
  --max_depth 8 \
  --learning_rate 0.1 \
  --random_state 42
```

Outputs:

```text
model.joblib
metrics.json
tensorboard/
```

## Vertex AI Custom Job

Edit `env.txt`, then run:

```bash
python scripts/submit_training_job.py --env_file env.txt
```

Use `--no_submit` to build/upload the package and print the job configuration without submitting it.
