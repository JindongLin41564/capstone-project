"""Build a source distribution for Vertex AI custom training."""

from setuptools import find_packages, setup


setup(
    name="project_completion_trainer",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "joblib==1.5.3",
        "numpy==1.26.4",
        "pandas==3.0.3",
        "scikit-learn==1.9.0",
        "tensorflow==2.17.1",
    ],
    description="GBDT project completion time prediction training package.",
)
