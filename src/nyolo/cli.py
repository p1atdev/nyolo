import argparse
import json
from pathlib import Path

from PIL import Image

from nyolo.checkpoint import load_checkpoint
from nyolo.inference import Predictor


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nyolo", description="Independent YOLO26-Pose inference"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--input", type=Path, nargs="+", required=True)
    predict.add_argument("--device", choices=("cpu", "mps"), required=True)
    predict.add_argument("--size", type=int, default=640)
    predict.add_argument("--score", type=float, default=0.25)
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint, device=args.device)
    predictor = Predictor(model, device=args.device, image_size=(args.size, args.size))
    images = []
    for path in args.input:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    result = predictor(images, score_threshold=args.score)
    document = {
        "coordinate_policy": model.config.coordinate_policy,
        "schema": model.config.schema.name,
        "inputs": [str(p) for p in args.input],
        "predictions": result.to_records(),
    }
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
