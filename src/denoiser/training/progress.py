import csv
import io

from tqdm import tqdm


def display(value, suffix=""):
    return "N/A" if value is None else f"{value}{suffix}"


class TrainingProgress:
    def __init__(self, directory, previous=None):
        self.path = directory / "metrics.csv"
        self.report = directory / "metrics.md"
        self.previous = previous

    def batch(self, bar, *, samples, total, loss, psnr, lr, grad, step):
        bar.set_postfix(samples=f"{samples}/{total}", loss=f"{loss:.5f}",
                        psnr=f"{psnr:.2f} dB", lr=f"{lr:.2g}", grad=f"{grad:.3f}", step=step)

    def epoch(self, epoch, samples, loss, psnr, lr, best, usage, eval_psnr=None, best_eval=False,
              best_train_score=None, best_eval_score=None):
        change = "first epoch"

        if self.previous is not None:
            delta = psnr - self.previous["psnr"]
            direction = "higher" if delta > 0 else "lower" if delta < 0 else "unchanged"
            change = f"PSNR {direction} ({delta:+.2f} dB), loss delta={loss - self.previous['loss']:+.5f}"

        tqdm.write(f"epoch {epoch}: patches={samples}, loss={loss:.5f}, PSNR={psnr:.2f} dB, "
                   f"lr={lr:.2g}; {change}; best train PSNR={best}; eval PSNR={display(eval_psnr)}; best eval={best_eval}")

        tqdm.write(f"time={usage['seconds']:.1f}s; RAM peak={usage['ram_peak_mib']} MiB; "
                   f"VRAM peak={usage['vram_allocated_peak_mib']} MiB; GPU={usage['gpu_util_mean_percent']}%")

        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")

        if not self.path.exists():
            writer.writerow(["epoch", "patches", "loss", "train_psnr_db", "lr", "is_best_train", "eval_psnr_db", "is_best_eval", "best_train_psnr_db", "best_eval_psnr_db", *usage])

        writer.writerow([epoch, samples, loss, psnr, lr, best, eval_psnr, best_eval, best_train_score, best_eval_score, *usage.values()])
        previous_text = self.path.read_text(encoding="utf-8") + "\n" if self.path.exists() else ""
        self.path.write_text(previous_text + stream.getvalue().rstrip("\n"), encoding="utf-8")
        section = (
            f"**Epoch {epoch}** — **{'BEST train PSNR' if best else 'training'}**; {change}\n\n"
            f"**Eval PSNR: {display(eval_psnr, ' dB')}**; **best eval: {best_eval}**\n\n"
            f"Best scores: train **{display(best_train_score, ' dB')}**, eval **{display(best_eval_score, ' dB')}**\n\n"
            "| Patches | Loss | PSNR | LR | Time |\n|---:|---:|---:|---:|---:|\n"
            f"| {samples} | **{loss:.5f}** | **{psnr:.2f} dB** | {lr:.2g} | **{usage['seconds']:.1f} s** |\n\n"
            "| RAM peak, sampled | VRAM allocated peak | VRAM reserved peak | GPU load, sampled mean |\n"
            "|---:|---:|---:|---:|\n"
            f"| {display(usage['ram_peak_mib'], ' MiB')} | **{display(usage['vram_allocated_peak_mib'], ' MiB')}** "
            f"| {display(usage['vram_reserved_peak_mib'], ' MiB')} | {display(usage['gpu_util_mean_percent'], '%')} |"
        )
        previous_report = self.report.read_text(encoding="utf-8") + "\n\n---\n\n" if self.report.exists() else (
            "Train PSNR uses stochastic training patches. Eval uses fixed noise on held-out center crops. "
            "RAM includes trainer and child processes; shared pages can be counted twice. "
            "GPU load covers the entire device; N/A means unavailable. Time includes saving images and checkpoints.\n\n"
        )
        self.report.write_text(previous_report + section, encoding="utf-8")
        self.previous = {"loss": loss, "psnr": psnr}