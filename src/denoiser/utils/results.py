from pathlib import Path

import yaml


def create_result_dir(cfg):
    """Reserve a new numbered run directory without overwriting earlier runs."""
    root = Path(cfg["data"]["result_dir"])
    root.mkdir(parents=True, exist_ok=True)
    layers = cfg["denoise"]["depth"] + 2
    channels = cfg["denoise"]["channels"]
    version = 1

    while True:
        path = root / f"layers_{layers}_channels_{channels}_v{version:03d}"

        try:
            path.mkdir()
        except FileExistsError:
            version += 1
            continue

        (path / "config.yaml").write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False).rstrip(), encoding="utf-8"
        )
        return path