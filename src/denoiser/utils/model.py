from pathlib import Path
import os
import tempfile

import torch
import yaml

from .errors import CheckpointError, CheckpointMessages


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path=None):
    if path is None:
        base_dir = Path(__file__).resolve().parents[1]
        path = base_dir / "config" / "project.yaml"

    with open(path) as f:
        cfg = yaml.safe_load(f)

    project_dir = Path(__file__).resolve().parents[3]

    for key in ("result_dir", "test_dir", "train_dir", "ckpt_dir"):
        cfg["data"][key] = str((project_dir / cfg["data"][key]).resolve())

    return cfg


def load_weights(model, cfg, device, *, allow_missing=False):
    path = Path(cfg["data"]["ckpt_dir"]) / cfg["data"]["denoise_path"]
    path = path.resolve()

    if not path.is_file():
        if allow_missing and not path.exists():
            print(CheckpointMessages.TRAINING_FROM_SCRATCH.format(path=path))
            return False

        raise CheckpointError(CheckpointMessages.NOT_FOUND.format(path=path))

    try:
        state = torch.load(
            path,
            weights_only=True,
            map_location=device
        )
        model.load_state_dict(state)
        print(f"Model weights loaded: {path}")
        return True

    except Exception as exc:
        raise CheckpointError(CheckpointMessages.LOAD_FAILED.format(path=path)) from exc


def save_weights(model, cfg):
    path = Path(cfg["data"]["ckpt_dir"]) / cfg["data"]["denoise_path"]
    atomic_save(model.state_dict(), path)


def atomic_save(state, path):
    path = Path(path)
    temporary = None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            torch.save(state, file)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary, path)
    except Exception as exc:
        raise CheckpointError(CheckpointMessages.SAVE_FAILED.format(path=path)) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)