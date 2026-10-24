"""Versioned tensor/primitive checkpoints; never load pickled model objects."""

from dataclasses import asdict
from pathlib import Path

import torch

from nyolo.learning.criterion import LossConfig, PoseTrainingModule
from nyolo.models.yolo26 import YOLO26Pose, YOLO26PoseConfig
from nyolo.nn.attention import AttentionBackend
from nyolo.pose.schema import PoseSchema
from nyolo.runtime import resolve_device


def _model_payload(model: YOLO26Pose) -> dict:
    if model.is_fused:
        raise ValueError("Save the original unfused model, then fuse after loading")
    return {
        "format_version": 1,
        "config": asdict(model.config),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }


def save_checkpoint(path: str | Path, model: YOLO26Pose) -> None:
    torch.save({**_model_payload(model), "kind": "model"}, path)


def save_training_state(path: str | Path, training: PoseTrainingModule) -> None:
    """Save model + criterion, including RLE. This is not a full trainer resume file."""
    torch.save(
        {
            **_model_payload(training.model),
            "kind": "training_components",
            "loss_config": asdict(training.criterion.config),
            "criterion_state": {
                k: v.detach().cpu() for k, v in training.criterion.state_dict().items()
            },
        },
        path,
    )


def _read(path: str | Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("format_version") != 1:
        raise ValueError("Unsupported nyolo checkpoint format")
    if data.get("kind") not in ("model", "training_components"):
        raise ValueError("Unsupported nyolo checkpoint kind")
    return data


def _restore_model(
    data: dict, device: str | torch.device, attention_backend: AttentionBackend
) -> YOLO26Pose:
    config = dict(data["config"])
    schema = dict(config.pop("schema"))
    for key in ("names", "flip_indices", "oks_sigmas"):
        schema[key] = tuple(schema[key])
    schema["skeleton"] = tuple(tuple(edge) for edge in schema["skeleton"])
    model = YOLO26Pose(
        YOLO26PoseConfig(schema=PoseSchema(**schema), **config),
        attention_backend=attention_backend,
    )
    model.load_state_dict(data["model_state"], strict=True)
    return model.to(resolve_device(device))


def load_checkpoint(
    path: str | Path, *, device: str, attention_backend: AttentionBackend = "sdpa"
) -> YOLO26Pose:
    return _restore_model(_read(path), device, attention_backend).eval()


def load_training_state(
    path: str | Path, *, device: str, attention_backend: AttentionBackend = "sdpa"
) -> PoseTrainingModule:
    data = _read(path)
    if data["kind"] != "training_components":
        raise ValueError("This checkpoint contains no criterion / RLE training state")
    result = PoseTrainingModule(
        _restore_model(data, device, attention_backend),
        LossConfig(**data["loss_config"]),
    )
    result.criterion.load_state_dict(data["criterion_state"], strict=True)
    return result.to(resolve_device(device))
