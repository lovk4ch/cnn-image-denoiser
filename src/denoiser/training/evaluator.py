import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from ..models.denoiser import Denoiser
from ..utils.common import get_psnr, img_to_tensor, create_result_dir
from ..utils.errors import DatasetMessages
from denoiser.training.weights import get_device, load_config, load_weights
from denoiser.training.noise import add_noise
from .metrics import DenoisingMetrics
from denoiser.diagnostics.training_log import quality_table
from denoiser.diagnostics.eval_image import make_example, sample_indices, save_eval_image


def evaluate(model, cfg, device, image_path=None):
    """Evaluate every noise mode with fixed CPU seeds; preserve training RNG and mode."""
    paths = sorted(p for p in Path(cfg["data"]["test_dir"]).iterdir()
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    if not paths:
        raise ValueError(DatasetMessages.NO_IMAGES.format(path=cfg["data"]["test_dir"]))

    overlap_allowed = cfg["eval"].get("allow_train_overlap", False)

    if overlap_allowed:
        tqdm.write("DIAGNOSTIC: train/test overlap is allowed; eval does not measure generalization.")

    if "train_dir" in cfg["data"] and not overlap_allowed:
        test_hashes = {hashlib.sha256(path.read_bytes()).digest() for path in paths}
        test_sizes = {path.stat().st_size for path in paths}

        for path in Path(cfg["data"]["train_dir"]).iterdir():
            if path.is_file() and path.stat().st_size in test_sizes:
                if hashlib.sha256(path.read_bytes()).digest() in test_hashes:
                    raise ValueError(DatasetMessages.TRAIN_EVAL_OVERLAP.format(path=path))

    was_training = model.training
    examples = []
    selected = sample_indices(len(paths))
    metrics = DenoisingMetrics()
    modes = tuple(cfg["eval"].get("modes", ("gaussian", "clean", "near_zero", "low", "iso", "mix", "hard")))

    if not modes or len(set(modes)) != len(modes) or not set(modes) <= {"gaussian", "clean", "near_zero", "low", "iso", "mix", "hard"}:
        raise ValueError(DatasetMessages.INVALID_EVAL_MODES)

    fingerprint = hashlib.sha256(json.dumps({"eval": cfg["eval"], "protocol": 2,
                                             "modes": modes, "noise": cfg.get("noise")}, sort_keys=True).encode())

    for path in paths:
        fingerprint.update(path.name.encode())
        fingerprint.update(path.read_bytes())

    model.eval()

    try:
        with torch.inference_mode():
            for index, path in enumerate(tqdm(paths, desc="eval", disable=not cfg["train"].get("live_progress", False)), 1):
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

                target = clean.to(device) * 2 - 1

                for mode in modes:
                    seed = cfg["eval"]["seed"] + int.from_bytes(
                        hashlib.sha256(f"{path.name}:{mode}".encode()).digest()[:4], "little")

                    with torch.random.fork_rng(devices=[]):
                        torch.random.default_generator.manual_seed(seed)

                        if mode == "gaussian" and "gaussian_sigma" not in cfg.get("noise", {}):
                            corrupted = (clean + torch.randn_like(clean) * cfg["eval"]["sigma"]).clamp(0, 1)
                        else:
                            noise_config = {"types": {mode: 100},
                                            "gaussian_sigma": cfg.get("noise", {}).get("gaussian_sigma", [0.03, 0.03])}
                            corrupted = add_noise(clean, noise_config)

                    noisy = corrupted.to(device) * 2 - 1
                    restored = (noisy - model(noisy)).clamp(-1, 1)
                    metrics.update(target, noisy, restored, [mode])
                    before, after = get_psnr(target, noisy), get_psnr(target, restored)

                    if image_path is not None and mode == "gaussian" and index in selected:
                        examples.append(make_example(path.name, target, noisy, restored, before, after,
                                                     image_size=cfg["eval"]["crop_size"]))
    finally:
        model.train(was_training)

    if image_path is not None:
        save_eval_image(examples, image_path)

    groups = metrics.summary()
    return {**groups["all"], "groups": groups, "images": len(paths), "diagnostic_overlap_allowed": overlap_allowed,
            "signature": fingerprint.hexdigest()}


def run_evaluation():
    cfg = load_config()
    device = get_device()
    model = Denoiser(cfg["denoise"]).to(device)
    load_weights(model, cfg, device)
    directory = create_result_dir(cfg)
    result = evaluate(model, cfg, device, directory / "eval_epoch.jpg")
    summary = (f"**Eval PSNR: {result['psnr']:.2f} dB**; "
               f"input: {result['input_psnr']:.2f} dB; gain: {result['gain']:+.2f} dB; images: {result['images']}")
    summary += "\n\n" + quality_table("eval", result["groups"])

    if result["diagnostic_overlap_allowed"]:
        summary = "**DIAGNOSTIC: train/test overlap allowed; this is not independent validation.**\n\n" + summary
    (directory / "eval_metrics.md").write_text(summary, encoding="utf-8")
    (directory / "eval_metrics.csv").write_text(
        f"eval_psnr_db,input_psnr_db,gain_db,images\n{result['psnr']},{result['input_psnr']},{result['gain']},{result['images']}", encoding="utf-8"
    )
    tqdm.write(summary)
    tqdm.write(f"Evaluation image: {directory / 'eval_epoch.jpg'}")
    tqdm.write(f"Evaluation metrics: {directory / 'eval_metrics.md'}")