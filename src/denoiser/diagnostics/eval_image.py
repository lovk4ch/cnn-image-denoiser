"""One labeled JPEG contact sheet for each evaluation epoch."""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..utils.common import tensor_to_img


MAX_EXAMPLES = 10
GAP = 12
LABEL_HEIGHT = 62
TITLE_HEIGHT = 56
FONT_SIZE = 18
BACKGROUND = "#0c1220"
TEXT = "#e5edf8"
ACCENT = "#8ea7ff"
JPEG_QUALITY = 95
DEFAULT_IMAGE_SIZE = 512


def sample_indices(count):
    limit = min(MAX_EXAMPLES, count)
    return {round(index * (count - 1) / max(1, limit - 1)) + 1 for index in range(limit)}


def label_font():
    for name in ("C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except OSError:
            continue

    return ImageFont.load_default(size=FONT_SIZE)


def make_example(filename, clean, noisy, restored, input_psnr, output_psnr, image_size=DEFAULT_IMAGE_SIZE):
    images = [tensor_to_img(tensor).convert("RGB") for tensor in (clean, noisy, restored)]
    width = height = image_size
    images = [ImageOps.pad(image, (width, height), method=Image.Resampling.BICUBIC, color=BACKGROUND)
              for image in images]
    row = Image.new("RGB", (3 * width + 4 * GAP, height + LABEL_HEIGHT + GAP), BACKGROUND)
    draw = ImageDraw.Draw(row)
    font = label_font()
    scores = (float("inf"), input_psnr, output_psnr)

    for index, (image, title, score) in enumerate(zip(images, ("Source", "Gaussian noise", "Correction"), scores)):
        left = GAP + index * (width + GAP)
        photo = f"Photo #{Path(filename).stem} | {title}"

        while draw.textlength(photo, font=font) > width and len(photo) > 4:
            photo = photo[:-4] + "..."

        display = "infinity (reference)" if math.isinf(score) else f"{score:.3f} dB"
        draw.text((left, 7), photo, font=font, fill=TEXT)
        draw.text((left, 31), f"PSNR: {display}", font=font, fill=ACCENT)
        row.paste(image, (left + (width - image.width) // 2, LABEL_HEIGHT))

    return row


def save_eval_image(examples, path):
    if not examples:
        return

    width = max(row.width for row in examples)
    sheet = Image.new("RGB", (width, TITLE_HEIGHT + sum(row.height for row in examples)), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    draw.text((GAP, 16), f"{path.stem} | Gaussian | {len(examples)} examples", font=label_font(), fill=TEXT)
    top = TITLE_HEIGHT

    for row in examples:
        sheet.paste(row, (0, top))
        top += row.height

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="JPEG", quality=JPEG_QUALITY, subsampling=0)