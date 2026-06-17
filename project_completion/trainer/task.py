"""Command-line entry point for local or Vertex AI custom training."""

import argparse

from trainer import model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--eval_data_path", required=True)
    parser.add_argument("--schema_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--label_scale", type=float, default=1.0)
    args = parser.parse_args()
    model.train_and_evaluate(vars(args))
