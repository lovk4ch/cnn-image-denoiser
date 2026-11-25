class DiagnosticsMessages:
    EVAL_IMAGE_MISSING = "No evaluation JPEG for this epoch. It is created after evaluation with the updated code."
    RUN_NOT_FOUND = "Result directory not found: {path}"
    INVALID_PORT = "port must be between 1 and 65535"
    FILE_CHANGING = "Metrics are being written; retry the next snapshot"
    LIBRARY_FAILED = "ECharts could not load. Check the connection or place echarts.min.js in diagnostics/vendor."
    CONNECTION_FAILED = "Chart server is unavailable; reconnecting"
    NOT_FOUND = "Not found"


class CheckpointMessages:
    SAVE_FAILED = "Failed to save checkpoint: {path}"
    CONFIG_MISMATCH = "Checkpoint configuration differs from the current training configuration"
    NOT_FOUND = "Checkpoint not found: {path}"
    LOAD_FAILED = "Failed to load checkpoint: {path}"
    TRAINING_FROM_SCRATCH = "Checkpoint not found; training from scratch: {path}"


class DatasetMessages:
    INVALID_EVAL_MODES = "eval.modes must be a nonempty list of unique supported noise types"
    INVALID_PATCH_COUNT = "patches_per_image must be positive"
    NO_IMAGES = "No images found in {path}"
    INVALID_IMAGE = "Invalid image in dataset: {path}"
    TRAIN_EVAL_OVERLAP = "The same image occurs in train and test: {path}. Use independent evaluation images."


class NoiseMessages:
    INVALID_PERCENTAGES = "Noise types must be gaussian, clean, near_zero, low, iso, mix or hard; percentages must be nonnegative and sum to 100"
    INVALID_SIGMA = "gaussian_sigma must contain two bounds with 0 <= minimum <= maximum <= 1"


class AuditMessages:
    INVALID_OPTIONS = "workers and grid must be positive; tile must be at least 32; skip must be nonnegative"
    NO_FILES = "No image files found for the grain audit"
    READ_FAILED = "Cannot inspect image: {detail}"
    MISSING_DIRECTORY = "Dataset directory does not exist: {path}"


class ModelMessages:
    INVALID_ACTIVATION = "activation must be relu or leaky_relu"
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