"""Run one baseline or all baselines in separate processes."""

import subprocess
import sys
from pathlib import Path
from baseline_cli import METHODS, parser, run_one


def main():
    args = parser().parse_args()
    if args.method != "all":
        run_one(args.method, args)
        return
    if not args.train:
        raise ValueError("--train is required when running all methods")
    if args.models and len(args.models) < 2:
        raise ValueError("Running all methods requires two models for Binoculars")
    root = Path(__file__).resolve().parent
    failures = []
    for method in METHODS:
        command = [
            sys.executable,
            str(root / "run_exp.py"),
            "--method",
            method,
            "--train",
            args.train,
            "--test",
            args.test,
            "--output",
            str(Path(args.output) / method),
            "--gpu",
            args.gpu,
        ]
        if args.task:
            command.extend(["--task", args.task])
        if args.models:
            models = (
                args.models[:1]
                if method == "lrr"
                else args.models[:2] if method == "binoculars" else args.models
            )
            command.extend(["--models", *models])
        for flag, value in [("--llama", args.llama), ("--qwen", args.qwen)]:
            if value:
                command.extend([flag, value])
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures.append(method)
    if failures:
        raise SystemExit("Failed methods: " + ", ".join(failures))


if __name__ == "__main__":
    main()
