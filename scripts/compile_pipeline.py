"""Compile the Vertex AI KFP pipeline from repo source."""

from kfp import compiler

from project_completion.pipeline_vertex.project_completion_kfp_pipeline import create_pipeline


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=create_pipeline,
        package_path="project_completion_kfp_pipeline.yaml",
    )
