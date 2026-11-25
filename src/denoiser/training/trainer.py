from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ImageDataset, collate_patches
from denoiser.diagnostics.training_log import TrainingProgress
from .resources import EpochResources
from .evaluator import evaluate
from .metrics import DenoisingMetrics
from denoiser.utils.paths import model_name
from ..models.denoiser import Denoiser
from denoiser.training.weights import save_weights, load_weights, load_config, get_device, atomic_save
from ..utils.errors import CheckpointError, CheckpointMessages
from denoiser.training.noise import add_noise
from ..utils.common import get_criterion, save_tensor_as_jpg, padding_cat, beep, \
    img_to_tensor, create_result_dir


class Trainer:
    def __init__(self):
        super().__init__()

        self.cfg = load_config() or {}
        self.device = get_device()
        self.loss = None

        self.model = Denoiser(self.cfg["denoise"]).to(self.device)
        self.criterion = get_criterion(self.cfg["train"])
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg["train"]["lr"])

        self.checkpoint_path = Path(self.cfg["data"]["ckpt_dir"]) / self.cfg["data"]["denoise_checkpoint"]
        self.start_epoch = 1
        self.step = 0
        self.best_loss = float("inf")
        self.best_train_psnr = float("-inf")
        self.best_eval_psnr = float("-inf")
        self.eval_signature = None
        self.previous_metrics = None

        if self.checkpoint_path.exists():
            try:
                state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)

                if model_name(state["config"]) != model_name(self.cfg):
                    raise ValueError(CheckpointMessages.CONFIG_MISMATCH)

                self.model.load_state_dict(state["model"])
                self.optim.load_state_dict(state["optimizer"])

                for group in self.optim.param_groups:
                    group["lr"] = self.cfg["train"]["lr"]

                self.start_epoch = state["epoch"] + 1
                self.step = state["step"]
                self.best_loss = state["best_loss"]
                self.best_train_psnr = state.get("best_train_psnr", float("-inf"))
                self.best_eval_psnr = state.get("best_eval_psnr", float("-inf"))
                self.eval_signature = state.get("eval_signature")
                self.previous_metrics = state.get("metrics")
                torch.set_rng_state(state["rng"].cpu())

                if self.device.type == "cuda" and "cuda_rng" in state:
                    torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
            except Exception as exc:
                raise CheckpointError(CheckpointMessages.LOAD_FAILED.format(path=self.checkpoint_path)) from exc
        else:
            load_weights(self.model, self.cfg, self.device, allow_missing=True)

        self.model.train()

    @property
    def grad(self):
        norms = [p.grad.detach().pow(2).sum() for p in self.model.parameters() if p.grad is not None]
        return torch.stack(norms).sum().sqrt().item() if norms else 0.0

    def __call__(self, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # +[0..1]
        with torch.no_grad():
            noisy, self.noise_modes = add_noise(batch, self.cfg.get("noise"), return_modes=True)

        # [-1..1]
        batch = (batch * 2 - 1)
        # [-1..1]
        noisy = (noisy * 2 - 1)

        base_noise = noisy - batch
        res_noise = self.model(noisy)
        clean = (noisy - res_noise).clamp(-1, 1)
        loss = self.criterion(base_noise, res_noise)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        self.optim.step()
        self.loss = loss.item()

        return batch, noisy, clean

    def run(self):
        # --- TRAIN ---

        train_dir = Path(self.cfg["data"]["train_dir"])
        train_dir.mkdir(parents=True, exist_ok=True)

        result_dir = create_result_dir(self.cfg)
        result_index = max((int(path.stem) for path in result_dir.glob("*.jpg") if path.stem.isdigit()), default=0) + 1
        progress = TrainingProgress(result_dir, self.previous_metrics,
                                    live=self.cfg["train"].get("live_progress", False),
                                    log_every=self.cfg["train"].get("log_every", 100),
                                    diagnostic=self.cfg["eval"].get("allow_train_overlap", False))

        train_dataset = ImageDataset(
            crop_size=self.cfg["train"]["train_crop"],
            root=train_dir,
            transform=img_to_tensor(),
            patches_per_image=self.cfg["train"].get("patches_per_image", 1),
        )

        train_loader = DataLoader(
            pin_memory=self.device.type == "cuda",
            batch_size=self.cfg["train"]["batch_size"],
            collate_fn=collate_patches,
            shuffle=True,
            num_workers=1,
            drop_last=False,
            dataset=train_dataset,
        )

        index = self.step + 1
        epochs = self.cfg["train"]["epochs"]
        result_every = self.cfg["train"]["result_every"]
        grad_every = max(1, self.cfg["train"].get("grad_every", 25))

        grad_norm = 0.0
        tqdm.write(f"Training photos: {len(train_dataset.paths)}; patches per epoch: {len(train_dataset)}")

        for epoch in range(self.start_epoch, epochs + 1):
            resources = EpochResources(self.device)
            resources.start()

            try:
                bar = tqdm(train_loader, desc=f"epoch {epoch}", disable=not progress.live)

                g_loss = 0
                samples = 0
                quality = DenoisingMetrics()

                for batch in bar:
                    batch = batch.to(self.device, non_blocking=True)
                    batch, noisy, clean = self(batch)

                    count = batch.size(0)

                    batch_quality = DenoisingMetrics()
                    batch_quality.update(batch, noisy, clean, self.noise_modes)
                    current = batch_quality.summary()["all"]
                    quality.update(batch, noisy, clean, self.noise_modes)
                    psnr = current["psnr"]

                    samples += count

                    g_loss += self.loss * count

                    if index == self.step + 1 or index % grad_every == 0:
                        grad_norm = self.grad

                    progress.batch(bar, samples=samples, total=len(train_dataset), loss=self.loss,
                                   psnr=psnr, lr=self.optim.param_groups[0]["lr"], grad=grad_norm, step=index,
                                   input_psnr=current["input_psnr"], gain=current["gain"],
                                   mode=",".join(sorted(set(self.noise_modes))))

                    if result_every > 0 and index % result_every == 0:
                        out = [batch, noisy, clean]
                        out = [t for t in out if t is not None]
                        rows = []

                        for i in range(batch.size(0)):
                            border = int(batch.size(3) * 10 // 1024)
                            row = padding_cat([t[i:i + 1] for t in out], border=border)
                            rows.append(row)

                        grid = torch.cat(rows, dim=2)
                        save_tensor_as_jpg(grid, result_dir / f"{result_index:05d}.jpg")
                        result_index += 1

                    index += 1

                print()

                g_loss /= samples
                train_groups = quality.summary()
                g_psnr = train_groups["all"]["psnr"]
                improved = g_psnr > self.best_train_psnr
                eval_metrics = evaluate(self.model, self.cfg, self.device, result_dir / f"eval_epoch_{epoch:05d}.jpg") if self.cfg["eval"]["enabled"] else None
                eval_psnr = eval_metrics["psnr"] if eval_metrics else None

                if eval_metrics and self.eval_signature != eval_metrics["signature"]:
                    self.best_eval_psnr = float("-inf")
                    self.eval_signature = eval_metrics["signature"]

                eval_improved = eval_psnr is not None and eval_psnr > self.best_eval_psnr
                self.best_train_psnr = max(self.best_train_psnr, g_psnr)

                if eval_improved:
                    self.best_eval_psnr = eval_psnr

                self.best_loss = min(self.best_loss, g_loss)
                self.step = index - 1
                state = {
                    "model": self.model.state_dict(), "optimizer": self.optim.state_dict(),
                    "epoch": epoch, "step": self.step, "best_loss": self.best_loss,
                    "config": self.cfg, "rng": torch.get_rng_state(),
                    "metrics": {"loss": g_loss, "psnr": g_psnr, "groups": train_groups},
                    "best_train_psnr": self.best_train_psnr, "best_eval_psnr": self.best_eval_psnr,
                    "eval_metrics": eval_metrics,
                    "eval_signature": self.eval_signature,
                }

                if self.device.type == "cuda":
                    state["cuda_rng"] = torch.cuda.get_rng_state_all()

                atomic_save(state, self.checkpoint_path)

                if improved:
                    atomic_save(state, self.checkpoint_path.with_name(f"{self.checkpoint_path.stem}_best_train.pth"))

                if eval_improved:
                    atomic_save(state, self.checkpoint_path.with_name(f"{self.checkpoint_path.stem}_best_eval.pth"))

                if eval_improved or (eval_psnr is None and improved):
                    save_weights(self.model, self.cfg)
            finally:
                usage = resources.finish()

            progress.epoch(epoch, samples, g_loss, g_psnr, self.optim.param_groups[0]["lr"], improved, usage,
                           eval_psnr=eval_psnr, best_eval=eval_improved,
                           best_train_score=self.best_train_psnr,
                           best_eval_score=self.best_eval_psnr if eval_psnr is not None else None,
                           train_groups=train_groups, eval_groups=eval_metrics["groups"] if eval_metrics else None)

        beep()