"""Command line for the sorAI random forest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sorai.rf import predict_from_probabilities


def _read_probability_table(path: Path) -> np.ndarray:
    table = pd.read_csv(path)
    if "prob_polyploid" not in table.columns:
        raise ValueError(f"{path} must contain a prob_polyploid column.")
    values = pd.to_numeric(table["prob_polyploid"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"{path} has no usable prob_polyploid values.")
    return values


def predict(args: argparse.Namespace) -> dict:
    if args.probabilities is not None:
        probabilities = _read_probability_table(Path(args.probabilities))
    else:
        from sorai.cnn import predict_loci

        probabilities = np.asarray(
            predict_loci(Path(args.loci_dir), Path(args.cnn_dir)),
            dtype=float,
        )
    return predict_from_probabilities(probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict diploid versus polyploid with the sorAI random forest."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["predict"],
        default="predict",
        help="Command to run. Only predict is supported.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--probabilities",
        help="CSV of per-locus probabilities with a prob_polyploid column.",
    )
    source.add_argument(
        "--loci-dir",
        help="Directory of tokenized locus alignments named L<probe_id>.json.",
    )
    parser.add_argument(
        "--cnn-dir",
        help="Directory of best_model_probe_3_<index>.pt checkpoints. Required with --loci-dir.",
    )
    parser.add_argument("--output", help="Optional JSON file for the result.")
    args = parser.parse_args()
    if args.loci_dir and not args.cnn_dir:
        parser.error("--cnn-dir is required with --loci-dir")

    result = predict(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
