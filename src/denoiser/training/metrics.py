import math

import torch


class DenoisingMetrics:
    """Accumulate pixel errors in [-1, 1]; PSNR is computed after averaging MSE."""

    def __init__(self):
        self.groups = {}

    def update(self, clean, noisy, restored, modes):
        with torch.no_grad():
            input_mse = (clean - noisy).square().flatten(1).mean(1).tolist()
            output_mse = (clean - restored).square().flatten(1).mean(1).tolist()

        for mode, before, after in zip(modes, input_mse, output_mse):
            for key in ("all", mode):
                group = self.groups.setdefault(key, [0, 0.0, 0.0])
                group[0] += 1
                group[1] += before
                group[2] += after

    def summary(self):
        result = {}

        for mode, (count, before, after) in self.groups.items():
            input_psnr = 10 * math.log10(4 / (before / count)) if before else float("inf")
            psnr = 10 * math.log10(4 / (after / count)) if after else float("inf")
            result[mode] = {"samples": count, "input_psnr": input_psnr, "psnr": psnr,
                            "gain": psnr - input_psnr if math.isfinite(input_psnr) and math.isfinite(psnr) else None,
                            "output_rmse_255": 127.5 * math.sqrt(after / count)}

        return result