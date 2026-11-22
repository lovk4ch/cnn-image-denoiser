import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from denoiser.utils.common import img_to_tensor
from denoiser.utils.errors import DatasetMessages


def collate_patches(images):
    """Crop each patch to the smallest side in this batch before stacking."""
    size = min(min(image.shape[-2:]) for image in images)
    patches = []

    for image in images:
        height, width = image.shape[-2:]
        top = torch.randint(height - size + 1, (1,)).item()
        left = torch.randint(width - size + 1, (1,)).item()
        patches.append(image[:, top:top + size, left:left + size])

    return torch.stack(patches)


class ImageDataset(Dataset):
    def __init__(self, root, crop_size=None, patches_per_image=1, transform=img_to_tensor()):
        if crop_size is None:
            crop_size = [256]

        self.paths = sorted(p for p in Path(root).iterdir()
                            if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})

        if not self.paths:
            raise ValueError(DatasetMessages.NO_IMAGES.format(path=root))

        for path in self.paths:
            try:
                with Image.open(path) as source:
                    source.load()
            except (OSError, ValueError) as exc:
                raise ValueError(DatasetMessages.INVALID_IMAGE.format(path=path)) from exc

        self.crop_size = crop_size

        if patches_per_image < 1:
            raise ValueError(DatasetMessages.INVALID_PATCH_COUNT)

        self.patches_per_image = patches_per_image
        self.transform = transform

    def __len__(self):
        return len(self.paths) * self.patches_per_image

    def __getitem__(self, index):
        path = self.paths[index % len(self.paths)]

        P = self.crop_size[torch.randint(0, len(self.crop_size), (1,)).item()]

        with Image.open(path) as source:
            image = source
            W, H = image.size

            if P > 0:
                if H < P or W < P:
                    scale = max(P / H, P / W)
                    image = image.resize((math.ceil(W * scale), math.ceil(H * scale)), Image.Resampling.BILINEAR)
                    W, H = image.size

                top = torch.randint(0, H - P + 1, (1,)).item()
                left = torch.randint(0, W - P + 1, (1,)).item()
                image = image.crop((left, top, left + P, top + P))

            return self.transform(image.convert('RGB'))