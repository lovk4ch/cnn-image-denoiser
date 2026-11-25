import csv
import io

from tqdm import tqdm


def display(value, suffix=""):
    return "N/A" if value is None else f"{value}{suffix}"


def quality_table(stage, groups):
    lines = [f"**{stage}: PSNR from mean pixel MSE**", "",
             "| Noise | Samples | Input PSNR | Output PSNR | Gain | Output RMSE /255 |",
             "|---|---:|---:|---:|---:|---:|"]

    for mode, values in groups.items():
        gain = "N/A" if values["gain"] is None else f"{values['gain']:+.3f}"
        lines.append(f"| {mode} | {values['samples']} | {values['input_psnr']:.3f} | "
                     f"**{values['psnr']:.3f}** | **{gain}** | {values['output_rmse_255']:.4f} |")

    return "\n".join(lines)


class TrainingProgress:
    def __init__(self, directory, previous=None, *, live=False, log_every=100, diagnostic=False):
        self.path = directory / "metrics.csv"
        self.report = directory / "metrics.md"
        self.previous = previous
        self.live = live
        self.log_every = max(1, log_every)
        self.diagnostic = diagnostic

    def batch(self, bar, *, samples, total, loss, psnr, lr, grad, step,
              input_psnr=None, gain=None, mode=""):
        values = dict(samples=f"{samples}/{total}", loss=f"{loss:.5f}",
                        psnr=f"{psnr:.2f}", input=f"{input_psnr:.2f}",
                        gain="N/A" if gain is None else f"{gain:+.2f}", noise=mode,
                        lr=f"{lr:.2g}", grad=f"{grad:.3g}", step=step)

        if self.live:
            bar.set_postfix(**values)
        elif samples == total or samples <= 1 or step % self.log_every == 0:
            print("batch: " + "; ".join(f"{key}={value}" for key, value in values.items()), flush=True)

    def epoch(self, epoch, samples, loss, psnr, lr, best, usage, eval_psnr=None, best_eval=False,
              best_train_score=None, best_eval_score=None, train_groups=None, eval_groups=None):
        change = "first epoch"

        if self.previous is not None:
            delta = psnr - self.previous["psnr"]
            direction = "higher" if delta > 0 else "lower" if delta < 0 else "unchanged"
            change = f"PSNR {direction} ({delta:+.2f} dB), loss delta={loss - self.previous['loss']:+.5f}"

        tqdm.write(f"epoch {epoch}: patches={samples}, loss={loss:.5f}, PSNR={psnr:.2f} dB, "
                   f"lr={lr:.2g}; {change}; best train PSNR={best}; "
                   f"eval PSNR={display(None if eval_psnr is None else round(eval_psnr, 3))}; best eval={best_eval}")

        tqdm.write(f"time={usage['seconds']:.1f}s; RAM peak={usage['ram_peak_mib']} MiB; "
                   f"VRAM peak={usage['vram_allocated_peak_mib']} MiB; GPU={display(usage['gpu_util_mean_percent'], '%')}")

        quality_stream = io.StringIO(newline="")
        quality_writer = csv.writer(quality_stream, lineterminator="\n")
        quality_path = self.path.with_name("quality.csv")

        if not quality_path.exists():
            quality_writer.writerow(["epoch", "stage", "noise", "samples", "input_psnr_db",
                                     "output_psnr_db", "gain_db", "output_rmse_255"])

        tables = []

        for stage, groups in (("train", train_groups), ("eval", eval_groups)):
            if not groups:
                continue

            tables.append(quality_table(stage, groups))

            for mode, values in groups.items():
                gain = "N/A" if values["gain"] is None else f"{values['gain']:+.3f} dB"
                tqdm.write(f"{stage}/{mode}: n={values['samples']}; input={values['input_psnr']:.3f} dB; "
                           f"output={values['psnr']:.3f} dB; gain={gain}; RMSE={values['output_rmse_255']:.4f}/255")
                quality_writer.writerow([epoch, stage, mode, values["samples"], values["input_psnr"],
                                         values["psnr"], values["gain"], values["output_rmse_255"]])

        previous_quality = quality_path.read_text(encoding="utf-8") + "\n" if quality_path.exists() else ""
        quality_path.write_text(previous_quality + quality_stream.getvalue().rstrip("\n"), encoding="utf-8")

        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        train_all = (train_groups or {}).get("all", {})
        eval_all = (eval_groups or {}).get("all", {})

        if not self.path.exists():
            writer.writerow(["epoch", "patches", "loss", "train_psnr_db", "lr", "is_best_train", "eval_psnr_db", "is_best_eval", "best_train_psnr_db", "best_eval_psnr_db",
                             "train_input_psnr_db", "train_gain_db", "eval_input_psnr_db", "eval_gain_db", *usage])

        writer.writerow([epoch, samples, loss, psnr, lr, best, eval_psnr, best_eval, best_train_score, best_eval_score,
                         train_all.get("input_psnr"), train_all.get("gain"),
                         eval_all.get("input_psnr"), eval_all.get("gain"), *usage.values()])
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

        if self.diagnostic:
            section = "**DIAGNOSTIC: train/test overlap allowed; eval does not measure generalization.**\n\n" + section

        section += "\n\n" + "\n\n".join(tables)
        previous_report = self.report.read_text(encoding="utf-8") + "\n\n---\n\n" if self.report.exists() else (
            "PSNR uses mean per-image MSE, then logarithm; values differ from older mean-dB reports. "
            "Train uses random patches/noise; eval tests configured equally weighted modes with fixed CPU noise seeds. "
            "Clean input has infinite PSNR: gain is N/A; output RMSE measures introduced distortion. "
            "RAM includes trainer and child processes; shared pages can be counted twice. "
            "GPU load covers the entire device; N/A means unavailable. Time includes saving images and checkpoints.\n\n"
        )
        self.report.write_text(previous_report + section, encoding="utf-8")
        self.previous = {"loss": loss, "psnr": psnr}