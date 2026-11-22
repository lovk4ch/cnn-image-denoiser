class CheckpointMessages:
    SAVE_FAILED = "Failed to save checkpoint: {path}"
    CONFIG_MISMATCH = "Checkpoint configuration differs from the current training configuration"
    NOT_FOUND = "Checkpoint not found: {path}"
    LOAD_FAILED = "Failed to load checkpoint: {path}"
    TRAINING_FROM_SCRATCH = "Checkpoint not found; training from scratch: {path}"


class DatasetMessages:
    INVALID_PATCH_COUNT = "patches_per_image must be positive"
    NO_IMAGES = "No images found in {path}"
    INVALID_IMAGE = "Invalid image in dataset: {path}"
    TRAIN_EVAL_OVERLAP = "The same image occurs in train and test: {path}. Use independent evaluation images."


class NoiseMessages:
    INVALID_PERCENTAGES = "Noise percentages must contain all six modes, be nonnegative and sum to 100"


class ModelMessages:
    INVALID_KERNEL = "kernel_size must be a positive odd integer"
    INVALID_DILATION = "dilation must contain exactly depth + 2 positive integers"


class ApiMessages:
    FILE_TOO_LARGE = "Image file exceeds {limit} MiB"
    IMAGE_TOO_LARGE = "Image dimensions exceed {limit} pixels per side"
    INVALID_IMAGE = "Invalid or damaged image file"
    UNSUPPORTED_IMAGE = "Supported formats: JPEG, PNG, WEBP and BMP; animations are not supported"
    BUSY = "The model is busy. Please try again shortly."


class CheckpointError(RuntimeError):
    """The model cannot be initialized from its checkpoint."""