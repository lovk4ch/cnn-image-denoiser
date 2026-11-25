from pathlib import Path


def model_name(cfg):
    model = cfg["denoise"]
    depth = int(model.get("depth", 5))
    dilation = model.get("dilation", 1)
    dilation = list(dilation) if isinstance(dilation, (list, tuple)) else [int(dilation)] * (depth + 2)
    pattern = "-".join(str(int(value)) for value in dilation)
    return f"layers_{depth + 2}_channels_{int(model.get('channels', 64))}_dilation_{pattern}"


def result_directory(cfg):
    return Path(cfg["data"]["result_dir"]) / model_name(cfg)