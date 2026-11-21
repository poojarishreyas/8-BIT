# Dummy train/test split

These files split the 275 non-system case IDs from the local Vajram evaluation report into 220 training and 55 test records. The split is stratified by the report's 27 pillars, using a stable SHA-256 ordering seeded with `arbok-dummy-v1`.

Each `prompt` and `completion` is synthetic placeholder text. The files are useful for checking a data-loading or evaluation pipeline, but they cannot measure meaningful model accuracy or loss. Replace them with real, held-out prompt/response data before evaluating a checkpoint.

The original report was `Vajram/docs/index.html` in the source clone; it is not included here.
