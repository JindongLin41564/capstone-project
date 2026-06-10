"""Data processing helpers for the project completion notebook.

This module contains the BigQuery SQL and export contract that used to live in
site_launch_prediction_main.ipynb. It is intentionally lightweight so the
notebook can run local smoke checks without GCP credentials.
"""

from dataclasses import dataclass


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


def build_preview_query(source_table: str) -> str:
    return f"""
SELECT
  RID,
  SID,
  PID,
  ZDP,
  GID,
  MID,
  S30,
  S44,
  S51,
  S52,
  S56,
  S68,
  S71,
  S90
FROM `{source_table}`
WHERE S30 IS NOT NULL
  AND S90 IS NOT NULL
"""


def build_feature_sql(source_table: str, feature_table: str) -> str:
    return f"""
CREATE OR REPLACE TABLE `{feature_table}` AS
SELECT
  DATE_DIFF(DATE(S90), DATE(S30), DAY) AS days_to_S90,
  COALESCE(CAST(RID AS STRING), 'UNKNOWN') AS RID,
  COALESCE(CAST(ZDP AS STRING), 'UNKNOWN') AS ZDP,
  COALESCE(CAST(GID AS STRING), 'UNKNOWN') AS GID,
  COALESCE(CAST(MID AS STRING), 'UNKNOWN') AS MID,
  DATE_DIFF(DATE(S44), DATE(S30), DAY) AS days_S44_from_S30,
  DATE_DIFF(DATE(S51), DATE(S30), DAY) AS days_S51_from_S30,
  DATE_DIFF(DATE(S52), DATE(S30), DAY) AS days_S52_from_S30,
  DATE_DIFF(DATE(S56), DATE(S30), DAY) AS days_S56_from_S30,
  DATE_DIFF(DATE(S68), DATE(S30), DAY) AS days_S68_from_S30,
  DATE_DIFF(DATE(S71), DATE(S30), DAY) AS days_S71_from_S30,
  COALESCE(CAST(SID AS STRING), '') AS SID,
  COALESCE(CAST(PID AS STRING), '') AS PID,
  'unused' AS key
FROM `{source_table}`
WHERE S30 IS NOT NULL
  AND S90 IS NOT NULL
  AND DATE_DIFF(DATE(S90), DATE(S30), DAY) >= 0
"""


def build_split_sql(feature_table: str, train_table: str, valid_table: str, test_table: str) -> str:
    split_expr = "ABS(MOD(FARM_FINGERPRINT(CONCAT(SID, '-', PID)), 100))"
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


def _bucket_and_prefix(gcs_uri: str) -> tuple[str, str]:
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gcs_uri}")
    path = gcs_uri.removeprefix("gs://")
    bucket, _, prefix = path.partition("/")
    return bucket, prefix.rstrip("/") + "/" if prefix else ""


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
    """Export train/valid/test tables as headerless CSV files."""

    from google.cloud import bigquery

    if clean_existing and storage_client is not None:
        bucket_name, prefix = _bucket_and_prefix(config.data_gcs_prefix)
        for blob in storage_client.list_blobs(bucket_name, prefix=prefix):
            blob.delete()

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
