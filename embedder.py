from transformers import AutoModel, AutoProcessor
import torch
import torch.nn.functional as F
import numpy as np
import math
import os
from PIL import Image
from contextlib import nullcontext
from dotenv import load_dotenv

load_dotenv()

model_id = os.getenv("BEDREFIN_MODEL_ID", "google/siglip2-base-patch16-256")
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
model = AutoModel.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
model.eval()
model = model.to(device)
autocast_dtype: torch.dtype | None = None
if device == "cuda":
    if torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float16
if autocast_dtype is not None:
    model = model.to(autocast_dtype)
if device == "cuda":
    model = torch.compile(model, dynamic=True)
EMBEDDING_DIM = int(model.config.text_config.hidden_size)

with torch.no_grad():
    _logit_scale = model.logit_scale.exp().item()
    _logit_bias = model.logit_bias.item()

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


def _autocast_context() -> torch.amp.autocast_mode.autocast | nullcontext:
    if device == "cuda" and autocast_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)
    return nullcontext()


def _to_embedding_tensor(outputs: object) -> torch.Tensor:
    """SigLIP get_*_features may return a Tensor or a HF output object."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    pooler = getattr(outputs, "pooler_output", None)
    if pooler is not None:
        return pooler
    last = getattr(outputs, "last_hidden_state", None)
    if last is not None:
        return last[:, 0, :]
    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")


def embed_text(text: str) -> np.ndarray:
    inputs = processor(text=[text], padding="max_length", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad(), _autocast_context():
        embedding = _to_embedding_tensor(model.get_text_features(**inputs))
    embedding = F.normalize(embedding, p=2, dim=-1)
    embedding = embedding.squeeze(0)
    embedding = embedding.cpu().numpy()
    return embedding


def embed_images(images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
    """Embed a list of images in batches. Returns an (N, D) array of L2-normalised embeddings."""
    if not images:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    all_embeddings = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        pixel_values = processor(images=batch_images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(device)
        with torch.no_grad(), _autocast_context():
            embeddings = _to_embedding_tensor(
                model.get_image_features(pixel_values=pixel_values)
            )
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)