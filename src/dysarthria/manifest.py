"""
Build a manifest (one row per utterance: path, label, speaker) from a
dataset directory, using rules from the YAML config:

    dataset:
      root: /data/TORGO
      labels:                      # path component -> class label
        F: dysarthric
        M: dysarthric
        FC: control
        MC: control
      label_from: speaker_prefix   # or: directory
      speaker_regex: '(?:^|/)((?:F|M|FC|MC)\\d+)(?:/|_)'

`speaker_regex` must capture the speaker ID in group 1. It is applied to
the path relative to `root`. Speaker IDs are what make speaker-independent
evaluation possible, so a file whose speaker cannot be parsed is an error,
not a silent skip.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Utterance:
    path: str
    label: str
    speaker: str


def _label_for(rel: str, speaker: str, labels: dict[str, str], label_from: str) -> str | None:
    if label_from == "speaker_prefix":
        prefix = re.match(r"[A-Za-z]+", speaker)
        return labels.get(prefix.group(0)) if prefix else None
    if label_from == "directory":
        for part in Path(rel).parts[:-1]:
            if part in labels:
                return labels[part]
        return None
    if label_from == "speaker":
        return labels.get(speaker)
    raise ValueError(f"unknown label_from: {label_from!r}")


def build_manifest(dataset_cfg: dict, extensions: tuple[str, ...] = (".wav",)) -> list[Utterance]:
    root = Path(dataset_cfg["root"])
    speaker_re = re.compile(dataset_cfg["speaker_regex"])
    labels = dataset_cfg["labels"]
    label_from = dataset_cfg.get("label_from", "directory")

    rows, unparsed = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in extensions or p.name.startswith("."):
            continue
        rel = p.relative_to(root).as_posix()
        m = speaker_re.search(rel)
        if not m:
            unparsed.append(rel)
            continue
        speaker = m.group(1)
        label = _label_for(rel, speaker, labels, label_from)
        if label is None:
            continue  # e.g. a session or speaker not used in this task
        rows.append(Utterance(rel, label, speaker))

    if unparsed:
        sample = ", ".join(unparsed[:5])
        raise ValueError(f"{len(unparsed)} files did not match speaker_regex (e.g. {sample})")
    if not rows:
        raise ValueError(f"no labeled audio found under {root}")
    return rows


def write_manifest(rows: list[Utterance], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "speaker"])
        for r in rows:
            w.writerow([r.path, r.label, r.speaker])


def read_manifest(path: Path) -> list[Utterance]:
    with path.open(newline="") as f:
        return [Utterance(r["path"], r["label"], r["speaker"]) for r in csv.DictReader(f)]


def summarize(rows: list[Utterance]) -> dict:
    by_label: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for r in rows:
        by_label.setdefault(r.label, set()).add(r.speaker)
        counts[r.label] = counts.get(r.label, 0) + 1
    return {lab: {"utterances": counts[lab], "speakers": sorted(spk)} for lab, spk in sorted(by_label.items())}
