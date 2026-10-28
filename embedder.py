import coremltools as ct
import numpy as np
import math
import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import atexit
import multiprocessing
from collections import deque
from collections.abc import Iterator
from itertools import count
from pathlib import Path
from queue import Empty
from typing import NamedTuple
from dotenv import load_dotenv
from image_preprocessing import initialize_worker, load_processor, prepare_paths

load_dotenv()

model_id = os.getenv("BEDREFIN_MODEL_ID", "google/siglip2-base-patch32-256")
device = "coreml"
INFERENCE_DTYPE = "float16"
compute_units = os.getenv("BEDREFIN_COREML_COMPUTE_UNITS", "ALL")
model_directory_name = (
    "coreml" if model_id == "google/siglip2-base-patch16-256"
    else f"coreml_{model_id.rsplit('/', 1)[-1].removeprefix('siglip2-')}"
)
model_dir = Path(os.getenv(
    "BEDREFIN_COREML_DIR",
    str(Path(__file__).resolve().parent / "benchmark_data" / model_directory_name),
))


def _load_model(filename: str):
    model = ct.models.MLModel(str(model_dir / filename), compute_units=ct.ComputeUnit[compute_units])
    if model.user_defined_metadata.get("model_id") != model_id:
        raise ValueError(f"Core ML model does not match {model_id}: {filename}")
    return model


image_models = {size: _load_model(f"vision_{size}.mlpackage") for size in (1, 32)}
text_model = _load_model("text_1.mlpackage")
processor = load_processor(model_id)
TEXT_SEQUENCE_LENGTH = int(text_model.get_spec().description.input[0].type.multiArrayType.shape[1])
EMBEDDING_DIM = int(text_model.user_defined_metadata["embedding_dim"])
_logit_scale = float(text_model.user_defined_metadata["logit_scale"])
_logit_bias = float(text_model.user_defined_metadata["logit_bias"])
_preprocessor_pool = None
_preprocessor_ready = None
_preprocessor_batch_ids = count()
PREPROCESS_WORKERS = int(os.getenv("BEDREFIN_PREPROCESS_WORKERS", "4"))

# Cosine similarity where sigmoid(logit_scale * sim + logit_bias) = 0.5.
MATCH_THRESHOLD = -_logit_bias / _logit_scale


def cosine_to_prob(cosine_sim: float) -> float:
    """Convert raw cosine similarity to calibrated match probability."""
    logit = _logit_scale * cosine_sim + _logit_bias
    return 1.0 / (1.0 + math.exp(-logit))


def prob_to_cosine(prob: float) -> float:
    """Convert target probability (0..1) to a cosine threshold."""
    clipped_prob = min(max(prob, 1e-6), 1.0 - 1e-6)
    logit = math.log(clipped_prob / (1.0 - clipped_prob))
    return (logit - _logit_bias) / _logit_scale


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def embed_text(text: str) -> np.ndarray:
    inputs = processor(
        text=[text], padding="max_length", truncation=True,
        max_length=TEXT_SEQUENCE_LENGTH, return_tensors="np",
    )
    embeddings = text_model.predict({"input_ids": inputs["input_ids"].astype(np.int32)})["embeddings"]
    return _normalize(embeddings)[0]


def embed_prepared_images(pixel_values: np.ndarray) -> np.ndarray:
    if not len(pixel_values):
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    outputs = []
    for start in range(0, len(pixel_values), 32):
        batch = np.ascontiguousarray(pixel_values[start:start + 32], dtype=np.float32)
        count = len(batch)
        if count <= 8:
            for image_index in range(count):
                outputs.append(image_models[1].predict({
                    "pixel_values": batch[image_index:image_index + 1],
                })["embeddings"])
        else:
            if count < 32:
                batch = np.pad(batch, ((0, 32 - count), (0, 0), (0, 0), (0, 0)))
            outputs.append(image_models[32].predict({"pixel_values": batch})["embeddings"][:count])
    return _normalize(np.concatenate(outputs))


class PreparedImageBatch(NamedTuple):
    pixel_values: np.ndarray
    decode_seconds: float
    preprocess_seconds: float


def submit_image_paths(
    paths: list[str | Path], skip_blank: bool = False,
):
    global _preprocessor_pool, _preprocessor_ready
    if PREPROCESS_WORKERS < 1:
        raise ValueError("BEDREFIN_PREPROCESS_WORKERS must be at least 1")
    if _preprocessor_pool is None:
        context = multiprocessing.get_context("spawn")
        _preprocessor_ready = context.Queue()
        _preprocessor_pool = ProcessPoolExecutor(
            max_workers=PREPROCESS_WORKERS, mp_context=context,
            initializer=initialize_worker, initargs=(_preprocessor_ready, model_id),
        )
        atexit.register(_preprocessor_ready.close)
        atexit.register(_preprocessor_pool.shutdown)
        startup = [_preprocessor_pool.submit(os.getpid) for _ in range(PREPROCESS_WORKERS)]
        for _ in startup:
            try:
                _preprocessor_ready.get(timeout=30)
            except Empty as error:
                raise RuntimeError("Image preprocessing worker did not initialize") from error
        for future in startup:
            future.result()
    batch_id = next(_preprocessor_batch_ids)
    future = _preprocessor_pool.submit(
        prepare_paths, paths, model_id, batch_id, skip_blank,
    )
    try:
        while _preprocessor_ready.get(timeout=30) != batch_id:
            pass
    except Empty as error:
        future.cancel()
        raise RuntimeError("Image preprocessing worker did not start") from error
    return future


def iter_prepared_image_batches(
    paths: list[str | Path], batch_size: int = 32, prefetch: bool = True
) -> Iterator[PreparedImageBatch]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not paths:
        return
    if not prefetch:
        for start in range(0, len(paths), batch_size):
            yield PreparedImageBatch(*prepare_paths(
                paths[start:start + batch_size], model_id, processor=processor,
            ))
        return
    pending = deque()

    def submit_batch(start: int) -> None:
        pending.append(submit_image_paths(paths[start:start + batch_size]))

    try:
        next_start = 0
        for _ in range(PREPROCESS_WORKERS):
            if next_start >= len(paths):
                break
            submit_batch(next_start)
            next_start += batch_size
        for start in range(0, len(paths), batch_size):
            prepared = PreparedImageBatch(*pending.popleft().result())
            if next_start < len(paths):
                submit_batch(next_start)
                next_start += batch_size
            yield prepared
    finally:
        for future in pending:
            future.cancel()


def embed_images(images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
    """Embed a list of images in batches. Returns an (N, D) array of L2-normalised embeddings."""
    if not images:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    all_embeddings = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        pixel_values = processor(images=batch_images, return_tensors="np")["pixel_values"]
        all_embeddings.append(embed_prepared_images(pixel_values))

    return np.concatenate(all_embeddings, axis=0)