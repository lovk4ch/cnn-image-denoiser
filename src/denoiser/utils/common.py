import sys
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn.functional import pad
from torchvision.transforms import transforms as T
from torchvision.transforms.v2.functional import resize, to_pil_image
from .paths import result_directory


def img_to_tensor(normalize=False, max_size=-1):
    """
    :param normalize: whether the image is converted to the range [-1, 1]
    :param max_size: shrink big images (for low-memory servers)
    """
    transforms = []

    if max_size > 0:
        transforms.append(
            T.Lambda(lambda img: resize_max_size(img, max_size))
        )

    transforms.append(T.ToTensor())

    if normalize:
        transforms.append(T.Normalize(mean=[0.5], std=[0.5]))

    return T.Compose(transforms)


def resize_max_size(img, max_size):
    w, h = img.size
    max_dim = max(w, h)

    if max_dim <= max_size:
        return img

    scale = max_size / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)

    return resize(img, [new_h, new_w])


def to_image(n, is_1x1=True):
    n = n.detach().cpu().squeeze(0)

    if is_1x1:
        n = (n + 1) / 2
        n = n.clamp(0, 1)

    return n


def tensor_to_img(n, is_1x1=True):
    n = to_image(n, is_1x1)
    return to_pil_image(n)


def save_tensor_as_jpg(n, name, is_1x1=True):
    tensor_to_img(n, is_1x1).save(name)
    return


def get_psnr(pred, target, max_val=2.0):
    """PSNR for tensors in [-1, 1]."""
    mse = torch.mean((pred - target) ** 2)
    return 10 * torch.log10(max_val ** 2 / mse).item()


def beep():
    if sys.platform.startswith('win'):
        import winsound
        winsound.PlaySound(
            str(Path("notify.wav").resolve()),
            winsound.SND_FILENAME
    )
    return


def padding_cat(images, border=10, value=1.0):
    """
    images: list of tensors [1, 3, H, W]
    return: [1, 3, H+2b, N*(W+2b)]
    """
    bordered = []

    for img in images:
        # pad = (left, right, top, bottom)
        img_b = pad(
            img,
            pad=(border, border, border, border),
            value=value
        )
        bordered.append(img_b)

    return torch.cat(bordered, dim=3)  # dim=3 → ширина


def get_criterion(cfg_loss):
    name = cfg_loss["loss"].lower()

    if name == "l1":
        return nn.L1Loss()

    if name == "mse":
        return nn.MSELoss()

    if name in ("huber", "smooth_l1"):
        # PyTorch: SmoothL1Loss ~= Huber с параметром beta (в новых версиях)
        delta = cfg_loss.get("huber_delta", 1.0)
        return nn.SmoothL1Loss(beta=delta)

    raise ValueError(f"Unknown loss: {name}")


def create_result_dir(cfg):
    """Reuse one results directory for layers, channels and dilation."""
    path = result_directory(cfg)
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False).rstrip(), encoding="utf-8"
    )
    return path