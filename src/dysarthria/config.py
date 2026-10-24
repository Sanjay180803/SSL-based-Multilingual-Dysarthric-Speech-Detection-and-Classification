from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with Path(path).open() as f:
        cfg = yaml.safe_load(f)
    for key in ("dataset", "model", "output_dir"):
        if key not in cfg:
            raise ValueError(f"{path}: missing top-level key {key!r}")
    return cfg
