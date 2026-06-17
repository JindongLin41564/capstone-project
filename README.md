# Project Completion GBDT

Minimal local and Vertex AI Custom Training project for project completion time prediction.

## Install

```bash
pip install -r requirements-local.txt
```

The sklearn model is saved with `joblib`, so the training and loading
environments must use the same scikit-learn/joblib versions. This project pins
those versions in both `requirements-local.txt` and `project_completion/setup.py`.
After installing or changing versions, restart the notebook kernel.

## Local Training

```bash
cd project_completion
python -m trainer.task \
  --train_data_path "gs://.../project_completion/train/project-train-*.csv" \
  --eval_data_path "gs://.../project_completion/valid/project-valid-*.csv" \
  --schema_path "gs://.../project_completion/schema/feature_schema.json" \
  --output_dir "gs://.../project_completion/local_gbdt_sklearn190_YYYYMMDD_HHMMSS" \
  --n_estimators 100 \
  --max_depth 5 \
  --learning_rate 0.05 \
  --random_state 42
```

Outputs:

```text
model.joblib
model_metadata.json
metrics.json
feature_importance_encoded.csv
feature_importance_by_original_feature.csv
tensorboard/
```

Do not reuse an old output directory such as `local_gbdt_test` unless you intend
to overwrite it. The notebook creates a fresh `LOCAL_OUTPUT_DIR` by default and
loads the model from that same directory. Loading first checks
`model_metadata.json`; old artifacts without metadata are rejected before
`joblib.load()` runs.

## Vertex AI Custom Job

Edit `env.txt`, then run:

```bash
python scripts/submit_training_job.py --env_file env.txt
```

Use `--no_submit` to build/upload the package and print the job configuration without submitting it.
