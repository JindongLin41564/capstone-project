"""Data processing helpers for Dataform-built project completion features.

Dataform owns feature engineering. This module only copies the final feature
table, creates deterministic train/valid/test splits, exports CSV files, and
writes the feature schema contract consumed by the trainer.
"""

import json
from dataclasses import dataclass


LABEL_COLUMN = "days_to_S90"
ID_COLUMNS = ["so_nr", "projekt_id"]
EXCLUDE_PREFIXES = ["meta_"]
CATEGORICAL_TYPES = {"STRING", "BOOL", "BOOLEAN"}
NUMERIC_TYPES = {"INT64", "INTEGER", "FLOAT64", "FLOAT", "NUMERIC", "BIGNUMERIC"}


@dataclass(frozen=True)
class DataProcessingConfig:
    project_id: str
    dataset_id: str = "jindong_lin"
    table_id: str = "project_data"
    bucket_uri: str = ""
    table_prefix: str = "project_completion"

    @property
    def source_table(self) -> str:
        return f"{self.project_id}.{self.dataset_id}.{self.table_id}"

    @property
    def feature_table(self) -> str:
        return f"{self.project_id}.{self.dataset_id}.{self.table_prefix}_features"

    @property
    def train_table(self) -> str:
        return f"{self.project_id}.{self.dataset_id}.{self.table_prefix}_train"

    @property
    def valid_table(self) -> str:
        return f"{self.project_id}.{self.dataset_id}.{self.table_prefix}_valid"

    @property
    def test_table(self) -> str:
        return f"{self.project_id}.{self.dataset_id}.{self.table_prefix}_test"

    @property
    def data_gcs_prefix(self) -> str:
        if not self.bucket_uri:
            return ""
        return f"{self.bucket_uri.rstrip('/')}/{self.table_prefix}"

    @property
    def schema_gcs_path(self) -> str:
        if not self.data_gcs_prefix:
            return ""
        return f"{self.data_gcs_prefix}/schema/feature_schema.json"


def build_preview_query(source_table: str) -> str:
    return f"""
SELECT *
FROM `{source_table}`
WHERE {LABEL_COLUMN} IS NOT NULL
  AND {LABEL_COLUMN} >= 0
LIMIT 10
"""


def build_feature_sql(source_table: str, feature_table: str) -> str:
    return f"""
CREATE OR REPLACE TABLE `{feature_table}` AS
SELECT *
FROM `{source_table}`
WHERE {LABEL_COLUMN} IS NOT NULL
  AND {LABEL_COLUMN} >= 0
"""


def build_split_sql(feature_table: str, train_table: str, valid_table: str, test_table: str) -> str:
    split_expr = (
        "ABS(MOD(FARM_FINGERPRINT(CONCAT("
        "COALESCE(CAST(so_nr AS STRING), ''), '-', "
        "COALESCE(CAST(projekt_id AS STRING), '')"
        ")), 100))"
    )
    return f"""
CREATE OR REPLACE TABLE `{train_table}` AS
SELECT * FROM `{feature_table}`
WHERE {split_expr} < 80;

CREATE OR REPLACE TABLE `{valid_table}` AS
SELECT * FROM `{feature_table}`
WHERE {split_expr} BETWEEN 80 AND 89;

CREATE OR REPLACE TABLE `{test_table}` AS
SELECT * FROM `{feature_table}`
WHERE {split_expr} >= 90;
"""


def build_export_jobs(config: DataProcessingConfig) -> list[tuple[str, str]]:
    if not config.data_gcs_prefix:
        raise ValueError("bucket_uri is required to build export jobs")
    return [
        (config.train_table, f"{config.data_gcs_prefix}/train/project-train-*.csv"),
        (config.valid_table, f"{config.data_gcs_prefix}/valid/project-valid-*.csv"),
        (config.test_table, f"{config.data_gcs_prefix}/test/project-test-*.csv"),
    ]


def build_feature_schema(table) -> dict:
    """Infer trainer feature roles from the BigQuery table schema."""

    categorical_features = []
    numeric_features = []
    id_columns = []
    excluded_columns = []
    csv_columns = []
    field_types = {}

    for field in table.schema:
        name = field.name
        field_type = field.field_type.upper()
        csv_columns.append(name)
        field_types[name] = field_type

        if name == LABEL_COLUMN:
            continue
        if name in ID_COLUMNS:
            id_columns.append(name)
            continue
        if any(name.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
            excluded_columns.append(name)
            continue
        if field_type in CATEGORICAL_TYPES:
            categorical_features.append(name)
            continue
        if field_type in NUMERIC_TYPES:
            numeric_features.append(name)
            continue
        excluded_columns.append(name)

    if LABEL_COLUMN not in csv_columns:
        raise ValueError(f"Feature table must contain label column: {LABEL_COLUMN}")
    if not categorical_features and not numeric_features:
        raise ValueError("Feature table has no supported categorical or numeric feature columns")

    return {
        "label": LABEL_COLUMN,
        "id_columns": id_columns,
        "excluded_columns": excluded_columns,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "csv_columns": csv_columns,
        "field_types": field_types,
    }


def _bucket_and_prefix(gcs_uri: str) -> tuple[str, str]:
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gcs_uri}")
    path = gcs_uri.removeprefix("gs://")
    bucket, _, prefix = path.partition("/")
    return bucket, prefix.rstrip("/") + "/" if prefix else ""


def _write_text_to_gcs(storage_client, gcs_uri: str, text: str) -> None:
    bucket_name, prefix = _bucket_and_prefix(gcs_uri)
    bucket = storage_client.bucket(bucket_name)
    bucket.blob(prefix.rstrip("/")).upload_from_string(text, content_type="application/json")


def run_bigquery_data_processing(bq_client, config: DataProcessingConfig, location: str) -> dict[str, int]:
    """Create feature/split tables and return row counts."""

    bq_client.query(build_feature_sql(config.source_table, config.feature_table), location=location).result()
    bq_client.query(
        build_split_sql(config.feature_table, config.train_table, config.valid_table, config.test_table),
        location=location,
    ).result()

    row_counts = {}
    for table in [config.feature_table, config.train_table, config.valid_table, config.test_table]:
        row_counts[table] = bq_client.get_table(table).num_rows
    return row_counts


def export_tables_to_gcs(
    bq_client,
    config: DataProcessingConfig,
    location: str,
    storage_client=None,
    clean_existing: bool = True,
) -> None:
    """Export train/valid/test tables and the schema contract used by training."""

    from google.cloud import bigquery
    from google.cloud import storage

    if clean_existing and storage_client is not None:
        bucket_name, prefix = _bucket_and_prefix(config.data_gcs_prefix)
        for blob in storage_client.list_blobs(bucket_name, prefix=prefix):
            blob.delete()
    if storage_client is None:
        storage_client = storage.Client(project=config.project_id)

    feature_schema = build_feature_schema(bq_client.get_table(config.feature_table))
    _write_text_to_gcs(
        storage_client,
        config.schema_gcs_path,
        json.dumps(feature_schema, indent=2, sort_keys=True),
    )

    job_config = bigquery.job.ExtractJobConfig(
        destination_format=bigquery.DestinationFormat.CSV,
        field_delimiter=",",
        print_header=False,
    )
    for table, destination_uri in build_export_jobs(config):
        bq_client.extract_table(
            table,
            destination_uri,
            job_config=job_config,
            location=location,
        ).result()
