import argparse
import os
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModel


class PoolingHead(torch.nn.Module):
    def __init__(self, head, batch_size):
        super().__init__()
        self.head = head
        self.batch_size = batch_size
        self.hidden_size = head.attention.embed_dim
        self.num_heads = head.attention.num_heads
        self.head_size = self.hidden_size // self.num_heads

    def forward(self, hidden_states):
        probe = self.head.probe.repeat(self.batch_size, 1, 1)
        projected = []
        for index, inputs in enumerate((probe, hidden_states, hidden_states)):
            start = index * self.hidden_size
            stop = start + self.hidden_size
            values = torch.nn.functional.linear(
                inputs, self.head.attention.in_proj_weight[start:stop],
                self.head.attention.in_proj_bias[start:stop],
            )
            projected.append(values.reshape(
                self.batch_size, -1, self.num_heads, self.head_size,
            ).transpose(1, 2))
        query, key, value = projected
        scores = (query * self.head_size ** -0.5) @ key.transpose(-2, -1)
        pooled = (scores.softmax(dim=-1) @ value).transpose(1, 2)
        pooled = pooled.reshape(self.batch_size, 1, self.hidden_size)
        pooled = self.head.attention.out_proj(pooled)
        return (pooled + self.head.mlp(self.head.layernorm(pooled)))[:, 0]


class VisionEncoder(torch.nn.Module):
    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values):
        return self.vision_model(pixel_values=pixel_values, return_dict=False)[1]


class TextEncoder(torch.nn.Module):
    def __init__(self, text_model):
        super().__init__()
        self.text_model = text_model

    def forward(self, input_ids):
        hidden_states = self.text_model.embeddings(input_ids=input_ids)
        hidden_states = self.text_model.encoder(
            inputs_embeds=hidden_states, attention_mask=None, return_dict=False,
        )[0]
        hidden_states = self.text_model.final_layer_norm(hidden_states)
        return self.text_model.head(hidden_states[:, -1, :])


def main() -> None:
    load_dotenv()
    model_id = os.getenv("BEDREFIN_MODEL_ID", "google/siglip2-base-patch32-256")
    model_directory_name = (
        "coreml" if model_id == "google/siglip2-base-patch16-256"
        else f"coreml_{model_id.rsplit('/', 1)[-1].removeprefix('siglip2-')}"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if args.output_dir is None:
        args.output_dir = Path(os.getenv(
            "BEDREFIN_COREML_DIR",
            str(Path(__file__).resolve().parent / "benchmark_data" / model_directory_name),
        ))
    source = AutoModel.from_pretrained(model_id, attn_implementation="eager").eval()
    size = source.config.vision_config.image_size
    if args.text:
        encoder = TextEncoder(source.text_model).eval()
        example = torch.zeros(1, source.config.text_config.max_position_embeddings, dtype=torch.long)
        input_name = "input_ids"
        input_dtype = np.int32
        filename = "text_1.mlpackage"
    else:
        encoder = VisionEncoder(source.vision_model).eval()
        example = torch.zeros(args.batch_size, 3, size, size)
        input_name = "pixel_values"
        input_dtype = np.float32
        filename = f"vision_{args.batch_size}.mlpackage"
    with torch.no_grad():
        reference = (source.text_model(input_ids=example).pooler_output.numpy()
                     if args.text else encoder(example).numpy())
        np.testing.assert_allclose(encoder(example).numpy(), reference, atol=1e-5, rtol=1e-4)
        if not args.text:
            source.vision_model.head = PoolingHead(source.vision_model.head, args.batch_size)
            np.testing.assert_allclose(encoder(example).numpy(), reference, atol=1e-5, rtol=1e-4)
        traced = torch.jit.trace(encoder, example, check_trace=False)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name=input_name, shape=example.shape, dtype=input_dtype)],
        outputs=[ct.TensorType(name="embeddings", dtype=np.float32)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    converted.user_defined_metadata["model_id"] = model_id
    converted.user_defined_metadata["batch_size"] = str(args.batch_size)
    converted.user_defined_metadata["embedding_dim"] = str(source.config.text_config.hidden_size)
    with torch.no_grad():
        converted.user_defined_metadata["logit_scale"] = str(source.logit_scale.exp().item())
        converted.user_defined_metadata["logit_bias"] = str(source.logit_bias.item())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / filename
    if destination.exists():
        raise FileExistsError(destination)
    converted.save(str(destination))
    print(f"Saved {destination}", flush=True)
    actual = converted.predict({input_name: example.numpy().astype(input_dtype)})["embeddings"]
    assert actual.shape == reference.shape and np.isfinite(actual).all()
    cosine = np.sum(reference * actual, axis=1) / (
        np.linalg.norm(reference, axis=1) * np.linalg.norm(actual, axis=1)
    )
    print({"minimum_conversion_cosine": float(cosine.min())}, flush=True)
    assert cosine.min() > 0.999


if __name__ == "__main__":
    main()