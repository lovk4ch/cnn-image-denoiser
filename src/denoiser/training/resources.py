import subprocess
import threading
import time

import psutil
import torch


class EpochResources:
    def __init__(self, device):
        self.device = device
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.ram_peak = 0
        self.gpu_samples = []

    def start(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self):
        while not self.stop.is_set():
            ram = 0

            try:
                processes = [self.process] + self.process.children(recursive=True)

                for process in processes:
                    try:
                        ram += process.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            self.ram_peak = max(self.ram_peak, ram)

            if self.device.type == "cuda":
                try:
                    index = self.device.index if self.device.index is not None else torch.cuda.current_device()
                    uuid = str(torch.cuda.get_device_properties(index).uuid)
                    result = subprocess.run(
                        ["nvidia-smi", f"--id={uuid}", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=2,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )

                    if result.returncode == 0:
                        self.gpu_samples.append(float(result.stdout.strip()))
                except (OSError, subprocess.TimeoutExpired, ValueError, AttributeError):
                    pass

            self.stop.wait(1)

    def finish(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        seconds = time.perf_counter() - self.started
        self.stop.set()
        self.thread.join()
        cuda = self.device.type == "cuda"
        return {
            "seconds": round(seconds, 3),
            "ram_peak_mib": round(self.ram_peak / 2 ** 20, 1),
            "vram_allocated_peak_mib": round(torch.cuda.max_memory_allocated(self.device) / 2 ** 20, 1) if cuda else None,
            "vram_reserved_peak_mib": round(torch.cuda.max_memory_reserved(self.device) / 2 ** 20, 1) if cuda else None,
            "gpu_util_mean_percent": round(sum(self.gpu_samples) / len(self.gpu_samples), 1) if self.gpu_samples else None,
        }