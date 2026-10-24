from copy import deepcopy

import pytest
import torch
from PIL import Image
from torch import nn

from nyolo import YOLO26Pose, YOLO26PoseConfig
from nyolo.checkpoint import load_checkpoint, save_checkpoint, save_training_state
from nyolo.inference import Predictor
from nyolo.learning import PoseTrainingModule


@pytest.mark.parametrize("scale", tuple("nsmlx"))
def test_fusion_preserves_outputs_and_original_state(device, scale):
    torch.manual_seed(17)
    model = YOLO26Pose(YOLO26PoseConfig(scale=scale)).to(device).eval()
    # Non-identity statistics/affine parameters catch missing bias and eps terms.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.uniform_(-0.3, 0.3)
                module.running_var.uniform_(0.5, 1.5)
                module.weight.uniform_(0.8, 1.2)
                module.bias.uniform_(-0.2, 0.2)
    original = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    fused = model.fuse()
    assert fused.is_fused and not model.is_fused
    assert not any(m.training for m in fused.modules())
    assert fused.attention_backend == model.attention_backend
    assert not any(isinstance(m, nn.BatchNorm2d) for m in fused.modules())
    assert not any(p.requires_grad for p in fused.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), original[name], rtol=0, atol=0)
    # Includes a shape change and a non-square image; no fixed-size grid cache.
    with torch.inference_mode():
        for shape in ((1, 3, 64, 96), (2, 3, 96, 64)):
            images = torch.rand(shape, device=device)
            actual, expected = fused(images), model(images)
            for a, b in zip(actual, expected, strict=True):
                if isinstance(a, torch.Tensor):
                    torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
                else:
                    assert a == b


def test_fusion_lifecycle_and_checkpoint(tmp_path, device):
    model = YOLO26Pose().to(device)
    with pytest.raises(ValueError, match="eval"):
        model.fuse()
    model.eval()
    fused = model.fuse()
    assert fused.eval() is fused
    with pytest.raises(ValueError, match="inference-only"):
        fused.train()
    with pytest.raises(ValueError, match="inference-only"):
        fused.forward_train(torch.rand(2, 3, 64, 64, device=device))
    with pytest.raises(ValueError, match="original unfused"):
        save_checkpoint(tmp_path / "invalid.pt", fused)
    with pytest.raises(ValueError, match="original unfused"):
        save_training_state(tmp_path / "invalid-training.pt", PoseTrainingModule(fused))
    assert not (tmp_path / "invalid.pt").exists()
    save_checkpoint(tmp_path / "model.pt", model)
    restored = load_checkpoint(tmp_path / "model.pt", device=device).fuse()
    twice = fused.fuse()
    for key, value in fused.state_dict().items():
        torch.testing.assert_close(value, twice.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    # Creating an inference copy must not disable training of the source.
    model.train().forward_train(
        torch.rand(2, 3, 64, 64, device=device)
    ).one_to_one.keypoint_offsets.sum().backward()
    assert next(model.backbone.parameters()).grad is not None


def test_predictor_fusion_is_explicit_and_errors_propagate(device, monkeypatch):
    model = YOLO26Pose().eval()
    baseline = Predictor(
        deepcopy(model), device=device, image_size=(64, 64), fuse=False
    )
    optimized = Predictor(model, device=device, image_size=(64, 64))
    assert optimized.model.is_fused and not model.is_fused
    image = Image.new("RGB", (93, 47), (90, 30, 140))
    a, b = optimized([image]), baseline([image])
    for name in ("boxes", "scores", "keypoints", "keypoint_scores"):
        torch.testing.assert_close(
            getattr(a, name), getattr(b, name), rtol=1e-3, atol=1e-4
        )
    assert torch.equal(a.valid, b.valid)

    def fail(self):
        raise RuntimeError("deliberate fusion failure")

    monkeypatch.setattr(YOLO26Pose, "fuse", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        Predictor(YOLO26Pose(), device=device)
    assert not Predictor(YOLO26Pose(), device=device, fuse=False).model.is_fused
