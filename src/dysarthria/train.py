"""
Train and evaluate SVM / MLP classifiers on extracted layer embeddings.

    python -m dysarthria.train --features runs/tdsc_wav2vec2 --layers 1 4 7 12 \
        --classifier svm mlp --seeds 0 1 2 3 4 --out runs/tdsc_wav2vec2/results.json

Splits
------
speaker (default)  StratifiedGroupKFold over speakers: no speaker appears in
                   both training and test folds. This is the protocol to
                   report.
utterance          Plain StratifiedKFold over utterances, as in the original
                   notebook's train_test_split. Kept only to measure how much
                   speaker leakage inflates accuracy.

Each (layer, classifier) is evaluated over every seed x fold; the report
gives mean and standard deviation of accuracy and macro-F1, per-class
precision/recall/F1, and the summed confusion matrix.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_features(features_dir: Path, layer: int):
    X = np.load(features_dir / f"layer_{layer:02d}.npy")
    with (features_dir / "index.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(X):
        raise ValueError(f"index.csv has {len(rows)} rows but layer_{layer:02d}.npy has {len(X)}")
    labels = np.array([r["label"] for r in rows])
    speakers = np.array([r["speaker"] for r in rows])
    return X, labels, speakers


def make_classifier(name: str, seed: int, n_classes: int):
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="linear"))
    if name == "mlp":
        from dysarthria.mlp import MLPClassifier

        return make_pipeline(StandardScaler(), MLPClassifier(seed=seed))
    raise ValueError(f"unknown classifier {name!r}")


def splits(split: str, y, groups, n_splits: int, seed: int):
    if split == "speaker":
        n_spk = min(len(set(groups[y == c])) for c in np.unique(y))
        if n_spk < n_splits:
            raise ValueError(
                f"speaker split needs at least {n_splits} speakers per class; smallest class has {n_spk}. "
                f"Lower --folds."
            )
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(y, y, groups)
    if split == "utterance":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(y, y)
    raise ValueError(f"unknown split {split!r}")


def evaluate(X, y, groups, classifier: str, split: str, seeds: list[int], n_splits: int) -> dict:
    classes = sorted(np.unique(y))
    accs, f1s = [], []
    y_true_all, y_pred_all = [], []
    for seed in seeds:
        for train_idx, test_idx in splits(split, y, groups, n_splits, seed):
            if split == "speaker":
                assert not set(groups[train_idx]) & set(groups[test_idx]), "speaker leaked across folds"
            clf = make_classifier(classifier, seed, len(classes))
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            accs.append(accuracy_score(y[test_idx], pred))
            f1s.append(f1_score(y[test_idx], pred, labels=classes, average="macro", zero_division=0))
            y_true_all.extend(y[test_idx])
            y_pred_all.extend(pred)

    p, r, f, s = precision_recall_fscore_support(y_true_all, y_pred_all, labels=classes, zero_division=0)
    return {
        "classifier": classifier,
        "split": split,
        "runs": len(accs),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": {"labels": classes, "matrix": confusion_matrix(y_true_all, y_pred_all, labels=classes).tolist()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--classifier", nargs="+", default=["svm", "mlp"], choices=["svm", "mlp"])
    parser.add_argument("--split", default="speaker", choices=["speaker", "utterance"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = []
    for layer in args.layers:
        X, y, groups = load_features(args.features, layer)
        for clf in args.classifier:
            res = {"layer": layer, **evaluate(X, y, groups, clf, args.split, args.seeds, args.folds)}
            report.append(res)
            print(f"layer {layer:2d}  {clf:3s}  {args.split}-split  "
                  f"acc {res['accuracy_mean']:.3f} ± {res['accuracy_std']:.3f}  "
                  f"macro-F1 {res['macro_f1_mean']:.3f} ± {res['macro_f1_std']:.3f}  ({res['runs']} runs)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
