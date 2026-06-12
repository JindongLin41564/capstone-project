"""Command-line entry point for Vertex AI custom training."""

import argparse

from trainer import model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--eval_data_path", required=True)
    parser.add_argument("--schema_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train_steps_per_epoch", type=int, required=True)
    parser.add_argument("--validation_steps", type=int, required=True)
    parser.add_argument("--hidden_units", default="128 64 32")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--label_scale", type=float, default=365.0)
    args = parser.parse_args()
    model.train_and_evaluate(vars(args))
