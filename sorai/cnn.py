"""Load a locus CNN and score one tokenized alignment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MAX_POSITIONS = 154
MAX_READS = 350


class DynamicLeNet(nn.Module):
    """Architecture stored inside the published locus checkpoints.

    Checkpoints were pickled as this class. ``torch.load`` restores the
    trained layers; ``forward`` has to match the class used when those
    checkpoints were evaluated.
    """

    def __init__(
        self,
        input_channels: int,
        input_size: int,
        num_layers: int,
        num_filters: list[int],
        kernel_sizes: list[int],
        dropout_rate: float,
        activation_fn: str,
        pool_type: str,
        fc_layer_sizes: list[int],
        num_classes: int = 1,
    ):
        super().__init__()
        layers = []
        in_channels = input_channels
        current_size = input_size

        for i in range(num_layers):
            out_channels = num_filters[i]
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_sizes[i],
                    stride=1,
                    padding=kernel_sizes[i] // 2,
                )
            )
            if activation_fn == "relu":
                layers.append(nn.ReLU())
            elif activation_fn == "leaky_relu":
                layers.append(nn.LeakyReLU())
            elif activation_fn == "elu":
                layers.append(nn.ELU())

            if pool_type == "max":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif pool_type == "avg":
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

            current_size = (current_size - kernel_sizes[i] + 2 * (kernel_sizes[i] // 2)) // 1 + 1
            current_size = current_size // 2
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((current_size, current_size))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_layers = None
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        if self.fc_layers:
            x = self.fc_layers(x)
        return self.sigmoid(x)


def locus_map_path() -> Path:
    return Path(__file__).resolve().parent / "models" / "probe_index_to_id.csv"


def load_locus_map(path: Path | None = None) -> dict[int, int]:
    table = pd.read_csv(path or locus_map_path())
    mapping = {}
    for row in table.itertuples(index=False):
        mapping[int(row.probe_id)] = int(row.model_index)
    return mapping


def _pad_row(tokens: list, length: int = MAX_POSITIONS) -> list[int]:
    row = [int(value) for value in tokens[:length]]
    if len(row) < length:
        row.extend([0] * (length - len(row)))
    return row


def alignment_tensor(reference_tokens: list, auxiliary_tokens: list) -> np.ndarray:
    """Return a float array of shape (1, 1, 351, 154)."""
    reference = _pad_row(reference_tokens)
    reads = [_pad_row(row) for row in auxiliary_tokens[:MAX_READS]]
    pad = [0] * MAX_POSITIONS
    if len(reads) < MAX_READS:
        reads.extend([pad] * (MAX_READS - len(reads)))
    matrix = np.asarray([reference, *reads], dtype=np.float32)
    if matrix.shape != (MAX_READS + 1, MAX_POSITIONS):
        raise ValueError(f"Expected shape {(MAX_READS + 1, MAX_POSITIONS)}, got {matrix.shape}")
    return matrix.reshape(1, 1, MAX_READS + 1, MAX_POSITIONS)


def read_locus_json(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    if "reference_tokens" not in payload:
        if len(payload) != 1:
            raise ValueError(f"{path} must contain reference_tokens or a single sample record.")
        payload = next(iter(payload.values()))
    return alignment_tensor(payload["reference_tokens"], payload["auxiliary_tokens"])


def load_cnn(checkpoint: Path, device: torch.device) -> nn.Module:
    import __main__

    __main__.DynamicLeNet = DynamicLeNet
    model = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(model, nn.Module):
        raise TypeError(f"{checkpoint} did not contain a network (got {type(model).__name__}).")
    model.to(device)
    model.eval()
    return model


def predict_locus(model: nn.Module, matrix: np.ndarray, device: torch.device) -> float:
    tensor = torch.tensor(matrix, dtype=torch.float32, device=device)
    with torch.no_grad():
        probability = model(tensor).detach().cpu().numpy().reshape(-1)
    return float(probability[0])


def checkpoint_path(cnn_dir: Path, model_index: int) -> Path:
    return cnn_dir / f"best_model_probe_3_{model_index}.pt"


def predict_loci(loci_dir: Path, cnn_dir: Path, device: torch.device | None = None) -> list[float]:
    """Score every L*.json file that has a matching locus checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping = load_locus_map()
    probabilities = []
    paths = sorted(Path(loci_dir).glob("L*.json"))
    if not paths:
        raise FileNotFoundError(f"No L*.json alignments were found in {loci_dir}.")
    for path in paths:
        probe_text = path.stem[1:]
        if not probe_text.isdigit():
            continue
        probe_id = int(probe_text)
        model_index = mapping.get(probe_id)
        if model_index is None:
            continue
        checkpoint = checkpoint_path(Path(cnn_dir), model_index)
        if not checkpoint.exists():
            continue
        model = load_cnn(checkpoint, device)
        probabilities.append(predict_locus(model, read_locus_json(path), device))
    if not probabilities:
        raise FileNotFoundError(
            "None of the locus files had a matching best_model_probe_3_<index>.pt checkpoint."
        )
    return probabilities
