"""
Smoke tests on synthetic data. The real corpora are licensed and are not
in the repository, so these test the pipeline's logic, not its accuracy.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dysarthria.extract import pool_layers
from dysarthria.manifest import build_manifest, read_manifest, summarize, write_manifest
from dysarthria.train import evaluate

ROOT = Path(__file__).resolve().parents[1]


def _touch(root: Path, rel: str):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


# --- manifest --------------------------------------------------------------

def test_torgo_style_manifest(tmp_path):
    for rel in ["F01/Session1/wav_arrayMic/0001.wav", "FC01/Session1/wav_arrayMic/0001.wav",
                "M02/Session2/wav_headMic/0003.wav", "MC03/Session1/wav_headMic/0002.wav"]:
        _touch(tmp_path, rel)
    rows = build_manifest({
        "root": str(tmp_path), "label_from": "speaker_prefix",
        "labels": {"F": "dysarthric", "M": "dysarthric", "FC": "control", "MC": "control"},
        "speaker_regex": r"^((?:FC|MC|F|M)\d+)/",
    })
    got = {(r.speaker, r.label) for r in rows}
    assert got == {("F01", "dysarthric"), ("FC01", "control"), ("M02", "dysarthric"), ("MC03", "control")}


def test_uaspeech_style_manifest(tmp_path):
    for rel in ["high/M05/M05_B1_C12_M3.wav", "very_low/F02/F02_B2_UW3_M5.wav", "control/CM01/CM01_B1_D2_M2.wav"]:
        _touch(tmp_path, rel)
    rows = build_manifest({
        "root": str(tmp_path), "label_from": "directory",
        "labels": {"very_low": "severe", "high": "mild", "control": "control"},
        "speaker_regex": r"(?:^|/)(C?[FM]\d+)_",
    })
    assert {(r.speaker, r.label) for r in rows} == {("M05", "mild"), ("F02", "severe"), ("CM01", "control")}


def test_unparseable_speaker_is_an_error(tmp_path):
    _touch(tmp_path, "mild/no_speaker_here.wav")
    with pytest.raises(ValueError, match="did not match speaker_regex"):
        build_manifest({"root": str(tmp_path), "labels": {"mild": "mild"}, "speaker_regex": r"(S\d+)"})


def test_manifest_roundtrip(tmp_path):
    for rel in ["mild/S1/a.wav", "severe/S2/b.wav"]:
        _touch(tmp_path / "data", rel)
    rows = build_manifest({"root": str(tmp_path / "data"), "labels": {"mild": "mild", "severe": "severe"},
                           "speaker_regex": r"^[^/]+/([^/]+)/"})
    write_manifest(rows, tmp_path / "m.csv")
    assert read_manifest(tmp_path / "m.csv") == rows
    assert summarize(rows)["mild"] == {"utterances": 1, "speakers": ["S1"]}


# --- feature pooling -------------------------------------------------------

def test_pool_layers_mean_pools_requested_layers():
    hs = [np.full((1, 5, 4), float(i)) for i in range(13)]  # 13 = encoder output + 12 layers
    out = pool_layers(hs, [1, 4, 12])
    assert set(out) == {1, 4, 12}
    assert out[4].shape == (4,) and np.allclose(out[4], 4.0)


def test_pool_layers_rejects_missing_layer():
    with pytest.raises(ValueError):
        pool_layers([np.zeros((1, 2, 3))] * 3, [5])


# --- evaluation ------------------------------------------------------------

def _speaker_confounded_data(seed=0, n_speakers=20, per_speaker=30, dim=16):
    """Each speaker has its own embedding centroid and a single label, and the
    label carries no signal beyond speaker identity. A classifier can only
    look good here by recognizing speakers it saw in training."""
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for s in range(n_speakers):
        centroid = rng.normal(0, 3, dim)
        label = "dysarthric" if s % 2 else "control"
        X.append(centroid + rng.normal(0, 0.3, (per_speaker, dim)))
        y += [label] * per_speaker
        g += [f"S{s:02d}"] * per_speaker
    return np.vstack(X), np.array(y), np.array(g)


def test_utterance_split_is_inflated_by_speaker_leakage():
    X, y, g = _speaker_confounded_data()
    leaky = evaluate(X, y, g, "svm", "utterance", seeds=[0], n_splits=5)
    honest = evaluate(X, y, g, "svm", "speaker", seeds=[0, 1], n_splits=5)
    assert leaky["accuracy_mean"] > 0.95       # memorizes speakers
    assert honest["accuracy_mean"] < 0.75      # near chance on unseen speakers


def test_speaker_split_learns_real_signal():
    X, y, g = _speaker_confounded_data()
    X = X.copy()
    X[y == "dysarthric", 0] += 20.0  # a feature that actually depends on the label
    res = evaluate(X, y, g, "svm", "speaker", seeds=[0], n_splits=5)
    assert res["accuracy_mean"] > 0.95
    assert res["confusion_matrix"]["labels"] == ["control", "dysarthric"]
    assert sum(map(sum, res["confusion_matrix"]["matrix"])) == len(y)


def test_mlp_classifier_runs():
    pytest.importorskip("torch")
    X, y, g = _speaker_confounded_data(n_speakers=10, per_speaker=10)
    X[y == "dysarthric", 0] += 20.0
    res = evaluate(X, y, g, "mlp", "speaker", seeds=[0], n_splits=5)
    assert res["runs"] == 5 and 0.0 <= res["accuracy_mean"] <= 1.0


def test_too_few_speakers_for_folds_is_reported():
    X, y, g = _speaker_confounded_data(n_speakers=4)
    with pytest.raises(ValueError, match="speakers per class"):
        evaluate(X, y, g, "svm", "speaker", seeds=[0], n_splits=5)


def test_train_cli_end_to_end(tmp_path):
    X, y, g = _speaker_confounded_data()
    np.save(tmp_path / "layer_04.npy", X)
    with (tmp_path / "index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "path", "label", "speaker"])
        for i, (lab, spk) in enumerate(zip(y, g)):
            w.writerow([i, f"{spk}/{i}.wav", lab, spk])
    out = tmp_path / "results.json"
    subprocess.run([sys.executable, "-m", "dysarthria.train", "--features", str(tmp_path), "--layers", "4",
                    "--classifier", "svm", "--seeds", "0", "--out", str(out)],
                   check=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""})
    report = json.loads(out.read_text())
    assert report[0]["layer"] == 4 and report[0]["split"] == "speaker"
