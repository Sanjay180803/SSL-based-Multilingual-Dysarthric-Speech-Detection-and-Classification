"""
Extract layer-wise utterance embeddings from a self-supervised speech model.

For every utterance in the manifest, the model's hidden states are
computed once; each requested layer is mean-pooled over time to a single
vector. Output: one ``layer_XX.npy`` matrix (n_utterances x hidden_size)
per layer, plus ``index.csv`` aligning rows with the manifest.

    python -m dysarthria.extract --config configs/tdsc_wav2vec2.yaml

Layer numbering follows the paper: layer 0 is the CNN feature encoder
output, layers 1..N are transformer layers.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from dysarthria.config import load_config
from dysarthria.manifest import build_manifest, write_manifest

TARGET_SR = 16_000


def pool_layers(hidden_states, layers: list[int]) -> dict[int, np.ndarray]:
    """hidden_states: sequence of (1, T, D) tensors/arrays, index = layer.
    Returns {layer: (D,) mean over time}."""
    out = {}
    for layer in layers:
        if layer >= len(hidden_states):
            raise ValueError(f"layer {layer} requested but model has {len(hidden_states) - 1} layers")
        h = hidden_states[layer]
        h = h.detach().cpu().numpy() if hasattr(h, "detach") else np.asarray(h)
        out[layer] = h.reshape(-1, h.shape[-1]).mean(axis=0)
    return out


def load_audio(path: Path):
    import torch
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)  # mono
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.to(torch.float32)


class Wav2Vec2Encoder:
    def __init__(self, checkpoint: str, device: str):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        self.device = device
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint)
        self.model = Wav2Vec2Model.from_pretrained(checkpoint).to(device).eval()

    def hidden_states(self, wav):
        import torch

        inputs = self.fe(wav.numpy(), sampling_rate=TARGET_SR, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model(**inputs, output_hidden_states=True).hidden_states


class DistilALHuBERTEncoder:
    """Wraps the UpstreamExpert from https://github.com/backspacetg/distilAlhubert.
    Clone that repo and set model.repo_path to it."""

    def __init__(self, checkpoint: str, device: str, repo_path: str):
        import sys

        sys.path.insert(0, repo_path)
        from src.upstream.alhubert.expert import UpstreamExpert  # type: ignore

        self.device = device
        self.model = UpstreamExpert(checkpoint).to(device).eval()

    def hidden_states(self, wav):
        import torch

        with torch.no_grad():
            return self.model([wav.to(self.device)])["hidden_states"]


def make_encoder(model_cfg: dict, device: str):
    kind = model_cfg["type"]
    if kind == "wav2vec2":
        return Wav2Vec2Encoder(model_cfg["checkpoint"], device)
    if kind == "distilalhubert":
        return DistilALHuBERTEncoder(model_cfg["checkpoint"], device, model_cfg["repo_path"])
    raise ValueError(f"unknown model type {kind!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if _cuda() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_manifest(cfg["dataset"])
    write_manifest(rows, out_dir / "manifest.csv")
    layers = cfg["model"]["layers"]
    encoder = make_encoder(cfg["model"], args.device)

    feats: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    root = Path(cfg["dataset"]["root"])
    for i, r in enumerate(rows, 1):
        pooled = pool_layers(encoder.hidden_states(load_audio(root / r.path)), layers)
        for layer, vec in pooled.items():
            feats[layer].append(vec)
        if i % 200 == 0 or i == len(rows):
            print(f"{i}/{len(rows)} utterances")

    for layer, vecs in feats.items():
        np.save(out_dir / f"layer_{layer:02d}.npy", np.stack(vecs))
    with (out_dir / "index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "path", "label", "speaker"])
        for i, r in enumerate(rows):
            w.writerow([i, r.path, r.label, r.speaker])
    print(f"Wrote {len(layers)} layer matrices for {len(rows)} utterances to {out_dir}")


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
