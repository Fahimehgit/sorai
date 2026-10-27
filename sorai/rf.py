"""Random forest that turns per-locus polyploid probabilities into one call."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

FEATURE_COLUMNS = [
    "n_models",
    "mean_prob",
    "median_prob",
    "vote_fraction",
    "logit_mean_prob",
    "confidence_weighted_prob",
    "adaptive_top_conf_mean",
    "trimmed_conf_mean_70pct",
    "std_prob",
    "min_prob",
    "max_prob",
    "q10_prob",
    "q25_prob",
    "q75_prob",
    "q90_prob",
    "iqr_prob",
    "frac_uncertain_045_055",
    "frac_high_07",
    "frac_low_03",
    "mean_abs_centered",
    "entropy_mean",
]


def bundle_dir() -> Path:
    return Path(__file__).resolve().parent / "models" / "rf"


def features_from_probabilities(probabilities: np.ndarray) -> dict[str, float]:
    """Summarize locus-level P(polyploid) values the way the forest was trained."""
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    p = p[np.isfinite(p)]
    if p.size == 0:
        raise ValueError("No locus probabilities were provided.")

    order = np.argsort(np.abs(p - 0.5))[::-1]
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    confidence = np.abs(p - 0.5)
    k_top = min(100, p.size)
    keep = max(1, int(np.ceil(p.size * 0.7)))
    q10, q25, q75, q90 = np.quantile(p, [0.1, 0.25, 0.75, 0.9])
    entropy = -np.mean(
        clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
    )
    weighted = (
        float(np.average(p, weights=confidence))
        if float(np.sum(confidence)) > 0
        else float(np.mean(p))
    )
    return {
        "n_models": float(p.size),
        "mean_prob": float(np.mean(p)),
        "median_prob": float(np.median(p)),
        "vote_fraction": float(np.mean(p >= 0.5)),
        "logit_mean_prob": float(
            1.0 / (1.0 + np.exp(-np.mean(np.log(clipped / (1.0 - clipped)))))
        ),
        "confidence_weighted_prob": weighted,
        "adaptive_top_conf_mean": float(np.mean(p[order[:k_top]])),
        "trimmed_conf_mean_70pct": float(np.mean(p[order[:keep]])),
        "std_prob": float(np.std(p)),
        "min_prob": float(np.min(p)),
        "max_prob": float(np.max(p)),
        "q10_prob": float(q10),
        "q25_prob": float(q25),
        "q75_prob": float(q75),
        "q90_prob": float(q90),
        "iqr_prob": float(q75 - q25),
        "frac_uncertain_045_055": float(np.mean((p >= 0.45) & (p <= 0.55))),
        "frac_high_07": float(np.mean(p >= 0.7)),
        "frac_low_03": float(np.mean(p <= 0.3)),
        "mean_abs_centered": float(np.mean(np.abs(p - 0.5))),
        "entropy_mean": float(entropy),
    }


def load_bundle(path: Path | None = None):
    root = Path(path) if path is not None else bundle_dir()
    with (root / "rf_model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    columns = json.loads((root / "feature_columns.json").read_text())
    metadata = json.loads((root / "model_metadata.json").read_text())
    if list(columns) != FEATURE_COLUMNS:
        raise ValueError("Feature column file does not match the sorAI random forest.")
    return model, columns, metadata


def predict_from_probabilities(
    probabilities: np.ndarray, bundle: Path | None = None
) -> dict:
    model, columns, metadata = load_bundle(bundle)
    feats = features_from_probabilities(probabilities)
    row = np.array([[feats[name] for name in columns]], dtype=float)
    probability = float(model.predict_proba(row)[0, 1])
    threshold = float(metadata["threshold"])
    finite = np.asarray(probabilities, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "prob_polyploid": probability,
        "call": "polyploid" if probability >= threshold else "diploid",
        "threshold": threshold,
        "n_loci": int(finite.size),
    }
