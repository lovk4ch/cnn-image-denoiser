import csv
import hashlib
import io
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from ..models.denoiser import Denoiser
from ..utils.common import get_psnr, img_to_tensor, padding_cat, save_tensor_as_jpg
from ..utils.errors import DatasetMessages
from ..utils.model import get_device, load_config, load_weights
from ..utils.results import create_result_dir


def evaluate(model, cfg, device, directory=None):
    """Fixed Gaussian noise and deterministic center crops; preserve training RNG and mode."""
    paths = sorted(p for p in Path(cfg["data"]["test_dir"]).iterdir()
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    if not paths:
        raise ValueError(DatasetMessages.NO_IMAGES.format(path=cfg["data"]["test_dir"]))

    if "train_dir" in cfg["data"]:
        test_hashes = {hashlib.sha256(path.read_bytes()).digest() for path in paths}
        test_sizes = {path.stat().st_size for path in paths}

        for path in Path(cfg["data"]["train_dir"]).iterdir():
            if path.is_file() and path.stat().st_size in test_sizes:
                if hashlib.sha256(path.read_bytes()).digest() in test_hashes:
                    raise ValueError(DatasetMessages.TRAIN_EVAL_OVERLAP.format(path=path))

    was_training = model.training
    rows = []
    fingerprint = hashlib.sha256(json.dumps(cfg["eval"], sort_keys=True).encode())

    for path in paths:
        fingerprint.update(path.name.encode())
        fingerprint.update(path.read_bytes())

    model.eval()

    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)

    try:
        with torch.inference_mode():
            for index, path in enumerate(tqdm(paths, desc="eval"), 1):
                try:
                    with Image.open(path) as source:
                        image = ImageOps.exif_transpose(source)
                        width, height = image.size
                        size = min(cfg["eval"]["crop_size"], width, height)
                        left, top = (width - size) // 2, (height - size) // 2
                        image = image.crop((left, top, left + size, top + size)).convert("RGB")
                        clean = img_to_tensor()(image).unsqueeze(0)
                except (OSError, ValueError) as exc:
                    raise ValueError(DatasetMessages.INVALID_IMAGE.format(path=path)) from exc

                seed = (cfg["eval"]["seed"] + int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:4], "little"))
                generator = torch.Generator().manual_seed(seed)
                noise = torch.randn(clean.shape, generator=generator) * cfg["eval"]["sigma"]
                noisy = (clean + noise).clamp(0, 1).to(device) * 2 - 1
                clean = clean.to(device) * 2 - 1
                restored = (noisy - model(noisy)).clamp(-1, 1)
                rows.append((path.name, get_psnr(clean, noisy), get_psnr(clean, restored)))

                if directory is not None:
                    grid = padding_cat([clean, noisy, restored])
                    save_tensor_as_jpg(grid, directory / f"{index:05d}.jpg")
    finally:
        model.train(was_training)

    if directory is not None:
        stream = io.StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["image", "input_psnr_db", "eval_psnr_db"])
        writer.writerows(rows)
        (directory / "images.csv").write_text(stream.getvalue().rstrip(), encoding="utf-8")

    return {"psnr": sum(row[2] for row in rows) / len(rows),
            "input_psnr": sum(row[1] for row in rows) / len(rows), "images": len(rows),
            "signature": fingerprint.hexdigest()}


def run_evaluation():
    cfg = load_config()
    device = get_device()
    model = Denoiser(cfg["denoise"]).to(device)
    load_weights(model, cfg, device)
    directory = create_result_dir(cfg)
    result = evaluate(model, cfg, device, directory / "eval")
    summary = (f"**Eval PSNR: {result['psnr']:.2f} dB**; "
               f"input: {result['input_psnr']:.2f} dB; images: {result['images']}")
    (directory / "metrics.md").write_text(summary, encoding="utf-8")
    (directory / "metrics.csv").write_text(
        f"eval_psnr_db,input_psnr_db,images\n{result['psnr']},{result['input_psnr']},{result['images']}", encoding="utf-8"
    )
    tqdm.write(summary)