"""Build a source distribution for Vertex AI custom training."""

from setuptools import find_packages, setup


setup(
    name="project_completion_trainer",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "joblib",
        "numpy",
        "pandas",
        "scikit-learn",
        "tensorflow",
    ],
    description="GBDT project completion time prediction training package.",
)
