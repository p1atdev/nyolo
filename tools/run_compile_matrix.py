"""Run named, independent compile probes sequentially (no backend fallback)."""

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument(
        "--scales", choices=tuple("nsmlx"), nargs="+", default=list("nsmlx")
    )
    parser.add_argument(
        "--backend", choices=("inductor", "aot_eager", "eager"), default="inductor"
    )
    parser.add_argument("--inference-size", type=int, default=640)
    parser.add_argument("--training-size", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [
        (
            scale,
            path,
            args.inference_size if path == "inference" else args.training_size,
        )
        for scale in args.scales
        for path in ("inference", "model_train")
    ]
    cases += [
        (args.scales[0], path, args.training_size)
        for path in ("inference_decode", "training_module")
    ]
    report = {
        "backend": args.backend,
        "device": args.device,
        "sequential": True,
        "results": [],
    }
    for scale, path, size in cases:
        name = f"{scale}-{path}"
        output = args.output / f"{name}.json"
        run_id = str(uuid.uuid4())
        command = [
            sys.executable,
            str(Path(__file__).with_name("probe_compile.py")),
            "--device",
            args.device,
            "--scale",
            scale,
            "--path",
            path,
            "--size",
            str(size),
            "--iterations",
            str(args.iterations),
            "--backend",
            args.backend,
            "--output",
            str(output),
            "--run-id",
            run_id,
        ]
        print(f"START {name} {size}x{size}", flush=True)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0"},
        )
        (args.output / f"{name}.log").write_text(completed.stdout + completed.stderr)
        if not output.exists():
            raise RuntimeError(
                f"Probe {name} exited {completed.returncode} without a result; see log"
            )
        result = json.loads(output.read_text())
        if result.get("run_id") != run_id:
            raise RuntimeError(
                f"Probe {name} did not write a result for the current run"
            )
        if bool(result["passed"]) != (completed.returncode == 0):
            raise RuntimeError(
                f"Probe {name} returned inconsistent status and exit code"
            )
        result["returncode"] = completed.returncode
        report["results"].append(result)
        (args.output / "matrix.json").write_text(
            json.dumps(report, allow_nan=False, indent=2) + "\n"
        )
        brief = {
            k: result[k]
            for k in (
                "scale",
                "path",
                "passed",
                "eager_median_ms",
                "compiled_median_ms",
                "speedup",
                "relative_gradient_l2_error",
                "error_type",
            )
            if k in result
        }
        print(json.dumps(brief), flush=True)
    # Boundary failures are part of the investigation, not successful execution.
    print(f"Completed {len(cases)} independent probes; inspect matrix.json", flush=True)


if __name__ == "__main__":
    main()
