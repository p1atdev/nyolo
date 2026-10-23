import json
import sys

import pytest
import torch
from PIL import Image

from nyolo import YOLO26Pose, YOLO26PoseConfig, cli
from nyolo.checkpoint import (
    load_checkpoint,
    load_training_state,
    save_checkpoint,
    save_training_state,
)
from nyolo.learning import PoseTrainingModule
from nyolo.nn import attention
from nyolo.nn.blocks import Attention
from nyolo.pose.types import PoseTargets


def assert_outputs_close(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
    elif isinstance(actual, tuple):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            assert_outputs_close(a, e)
    else:
        assert actual == expected


@pytest.mark.parametrize("channels", (128, 256, 384))
@pytest.mark.parametrize("training", (False, True))
def test_sdpa_matches_naive_outputs_and_gradients(device, channels, training):
    torch.manual_seed(19)
    naive = Attention(channels, attention_backend="naive").to(device).train(training)
    sdpa = Attention(channels, attention_backend="sdpa").to(device).train(training)
    sdpa.load_state_dict(naive.state_dict(), strict=True)
    x = torch.randn(2, channels, 7, 11, device=device, requires_grad=True)
    other_x = x.detach().clone().requires_grad_()
    expected, actual = naive(x), sdpa(other_x)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    upstream = torch.randn_like(expected)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    torch.testing.assert_close(other_x.grad, x.grad, rtol=1e-3, atol=1e-4)
    for (name, a), (expected_name, e) in zip(
        sdpa.named_parameters(), naive.named_parameters(), strict=True
    ):
        assert name == expected_name
        assert a.grad is not None and e.grad is not None, name
        torch.testing.assert_close(a.grad, e.grad, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("scale", tuple("nsmlx"))
def test_every_scale_uses_selected_backend_and_retains_state(device, scale):
    torch.manual_seed(7)
    config = YOLO26PoseConfig(scale=scale)
    naive = YOLO26Pose(config, attention_backend="naive").to(device).eval()
    sdpa = YOLO26Pose(config).to(device).eval()
    sdpa.load_state_dict(naive.state_dict(), strict=True)
    for model, backend in ((naive, "naive"), (sdpa, "sdpa")):
        blocks = [m for m in model.modules() if isinstance(m, Attention)]
        assert len(blocks) == config.repeats + 1
        assert model.attention_backend == backend
        assert all(m.attention_backend == backend for m in blocks)
    with torch.inference_mode():
        images = torch.rand(1, 3, 64, 96, device=device)
        assert_outputs_close(sdpa(images), naive(images))
        assert_outputs_close(sdpa.forward_train(images), naive.forward_train(images))


def test_real_training_loss_and_gradients_agree(device):
    torch.manual_seed(7)
    naive = PoseTrainingModule(YOLO26Pose(attention_backend="naive")).to(device)
    sdpa = PoseTrainingModule(YOLO26Pose(attention_backend="sdpa")).to(device)
    sdpa.load_state_dict(naive.state_dict(), strict=True)
    images = torch.rand(2, 3, 96, 128, device=device)
    points = torch.full((2, 1, 17, 2), 40.0, device=device)
    points[:, :, 0, 0] = -10.0
    visibility = torch.full((2, 1, 17), 2, dtype=torch.int64, device=device)
    visibility[:, :, 0] = 1
    visibility[:, :, 1] = 0
    points[:, :, 1] = float("nan")
    targets = PoseTargets(
        torch.tensor([[[16.0, 16.0, 80.0, 80.0]]] * 2, device=device),
        torch.zeros(2, 1, dtype=torch.int64, device=device),
        points,
        visibility,
        torch.ones(2, 1, dtype=torch.bool, device=device),
    )
    expected = naive(images, targets, progress=0.5).total
    actual = sdpa(images, targets, progress=0.5).total
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
    expected.backward()
    actual.backward()
    squared_error = squared_reference = 0.0
    for (name, a), (expected_name, e) in zip(
        sdpa.named_parameters(), naive.named_parameters(), strict=True
    ):
        assert name == expected_name
        assert a.grad is not None and e.grad is not None, name
        a_grad, e_grad = a.grad.cpu(), e.grad.cpu()
        assert torch.isfinite(a_grad).all() and torch.isfinite(e_grad).all(), name
        squared_error += float((a_grad - e_grad).square().sum())
        squared_reference += float(e_grad.square().sum())
    assert squared_reference > 0
    assert (squared_error / squared_reference) ** 0.5 < 0.005


@pytest.mark.parametrize("backend", ("naive", "sdpa"))
def test_checkpoint_backend_is_a_load_time_choice(tmp_path, device, backend):
    torch.manual_seed(11)
    training = PoseTrainingModule(YOLO26Pose(attention_backend=backend)).to(device)
    training.eval()
    path = tmp_path / "model.pt"
    training_path = tmp_path / "training.pt"
    save_checkpoint(path, training.model)
    save_training_state(training_path, training)
    payload = torch.load(path, weights_only=True)
    assert payload["format_version"] == 1
    assert "attention_backend" not in payload["config"]
    assert load_checkpoint(path, device=device).attention_backend == "sdpa"
    assert (
        load_training_state(training_path, device=device).model.attention_backend
        == "sdpa"
    )
    with torch.inference_mode():
        images = torch.rand(1, 3, 64, 96, device=device)
        expected = training.model(images)
        for chosen in ("naive", "sdpa"):
            restored = load_checkpoint(path, device=device, attention_backend=chosen)
            restored_training = load_training_state(
                training_path, device=device, attention_backend=chosen
            ).eval()
            for model in (restored, restored_training.model):
                assert model.attention_backend == chosen
                actual = model(images)
                assert_outputs_close(actual, expected)
                if chosen == backend:
                    torch.testing.assert_close(
                        actual.keypoint_offsets,
                        expected.keypoint_offsets,
                        rtol=0,
                        atol=0,
                    )
            for name, value in training.state_dict().items():
                torch.testing.assert_close(
                    restored_training.state_dict()[name], value, rtol=0, atol=0
                )


@pytest.mark.parametrize("backend", ("flash_attn", "auto", "typo", None))
def test_unknown_attention_backend_is_rejected(backend):
    with pytest.raises(ValueError, match="Unsupported attention backend"):
        YOLO26Pose(attention_backend=backend)
    with pytest.raises(ValueError, match="Unsupported attention backend"):
        Attention(128, attention_backend=backend)


def test_sdpa_errors_propagate_without_retry(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("SDPA deliberately unavailable")

    monkeypatch.setattr(attention.F, "scaled_dot_product_attention", fail)
    images = torch.rand(1, 128, 3, 5)
    with pytest.raises(RuntimeError, match="SDPA deliberately unavailable"):
        Attention(128, attention_backend="sdpa").eval()(images)
    assert torch.isfinite(
        Attention(128, attention_backend="naive").eval()(images)
    ).all()


@pytest.mark.parametrize("backend", (None, "naive", "sdpa"))
def test_cli_selects_attention_backend(tmp_path, monkeypatch, backend):
    path = tmp_path / "model.pt"
    image_path, output_path = tmp_path / "image.png", tmp_path / "prediction.json"
    save_checkpoint(path, YOLO26Pose(attention_backend="naive"))
    Image.new("RGB", (96, 64), (70, 100, 130)).save(image_path)
    loaded = []

    def load(*args, **kwargs):
        model = load_checkpoint(*args, **kwargs)
        loaded.append(model.attention_backend)
        return model

    monkeypatch.setattr(cli, "load_checkpoint", load)
    args = [
        "nyolo",
        "predict",
        "--checkpoint",
        str(path),
        "--input",
        str(image_path),
        "--device",
        "cpu",
        "--size",
        "64",
        "--output",
        str(output_path),
    ]
    if backend is not None:
        args.extend(("--attention-backend", backend))
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    assert loaded == ["sdpa" if backend is None else backend]
    assert json.loads(output_path.read_text())["coordinate_policy"] == "unbounded-v1"
