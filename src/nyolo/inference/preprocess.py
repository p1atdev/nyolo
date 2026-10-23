import numpy as np
import torch
from PIL import Image
from torch import Tensor

from nyolo.geometry.transforms import ImageTransform


def letterbox(
    image: Image.Image, size: tuple[int, int] = (640, 640)
) -> tuple[Tensor, ImageTransform]:
    """RGB / 255, bilinear resize, centered 114 padding; returns a CPU CHW tensor."""
    ih, iw = size
    if min(ih, iw) < 32 or ih % 32 or iw % 32:
        raise ValueError("Input dimensions must be multiples of 32")
    ow, oh = image.size
    ratio = min(iw / ow, ih / oh)
    rw, rh = max(1, round(ow * ratio)), max(1, round(oh * ratio))
    px, py = (iw - rw) // 2, (ih - rh) // 2
    canvas = Image.new("RGB", (iw, ih), (114, 114, 114))
    canvas.paste(
        image.convert("RGB").resize((rw, rh), Image.Resampling.BILINEAR), (px, py)
    )
    array = np.array(canvas, dtype=np.float32) / 255.0
    return (
        torch.from_numpy(array).permute(2, 0, 1).contiguous(),
        ImageTransform((oh, ow), size, (rh, rw), (px, py)),
    )
