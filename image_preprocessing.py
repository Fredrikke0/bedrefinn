import signal
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from transformers import AutoTokenizer, PreTrainedConfig, SiglipImageProcessorPil, SiglipProcessor

_processor = None
_ready = None


def load_processor(model_id: str, local_files_only: bool = False):
    image_processor = SiglipImageProcessorPil.from_pretrained(
        model_id, local_files_only=local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, config=PreTrainedConfig(), local_files_only=local_files_only,
    )
    config, kwargs = SiglipProcessor.get_processor_dict(
        model_id, local_files_only=local_files_only,
    )
    return SiglipProcessor.from_args_and_dict([image_processor, tokenizer], config, **kwargs)


def initialize_worker(ready, model_id: str):
    global _ready, _processor
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _ready = ready
    _processor = load_processor(model_id, local_files_only=True)
    _ready.put(None)


def prepare_paths(
    paths: list[str | Path], model_id: str, batch_id: int | None = None,
    skip_blank: bool = False, *, processor=None,
):
    global _processor
    if _ready is not None:
        _ready.put(batch_id)
    if processor is None:
        if _processor is None:
            _processor = load_processor(model_id, local_files_only=True)
        processor = _processor
    images = []
    kept_indices = []
    blank_indices = []
    decode_errors = []
    started = perf_counter()
    try:
        for index, path in enumerate(paths):
            try:
                with Image.open(path) as image:
                    converted = image.convert("RGB")
            except (OSError, ValueError) as error:
                if not skip_blank:
                    raise
                decode_errors.append(f"Error decoding image {path}: {error}")
                continue
            if skip_blank:
                with converted.convert("L") as grayscale:
                    blank = float(np.array(grayscale).std()) < 8.0
                if blank:
                    converted.close()
                    blank_indices.append(index)
                    continue
            images.append(converted)
            kept_indices.append(index)
        decode_seconds = perf_counter() - started
        started = perf_counter()
        pixel_values = (
            processor(images=images, return_tensors="np")["pixel_values"]
            if images else np.empty((0,), dtype=np.float32)
        )
        if skip_blank:
            return (
                pixel_values, decode_seconds, perf_counter() - started,
                kept_indices, blank_indices, decode_errors,
            )
        return pixel_values, decode_seconds, perf_counter() - started
    finally:
        for image in images:
            image.close()