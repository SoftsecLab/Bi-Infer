"""Shared paths, task names and arbitrary causal-LM checkpoints for baselines."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path

METHODS = {
    "binoculars": "Binoculars",
    "ghostbuster": "GhostBuster",
    "profiler": "Profiler",
    "gpt_who": "gpt_who",
    "lrr": "lrr",
    "seqxgpt": "seqxgpt",
    "sniffer": "sniffer",
}
SUPERVISED = {"profiler", "seqxgpt", "sniffer", "ghostbuster", "gpt_who"}


def parser(method=None):
    p = argparse.ArgumentParser(description=__doc__)
    if method is None:
        p.add_argument("--method", choices=[*METHODS, "all"], required=True)
    p.add_argument("--train", "--train_data", dest="train")
    p.add_argument("--test", "--test_data", dest="test", required=True)
    p.add_argument("--task", help="Any task name; defaults to the test filename stem")
    p.add_argument(
        "--models",
        nargs="+",
        help="Ordered local model paths or Hugging Face model IDs",
    )
    p.add_argument("--output", "--output_dir", dest="output", default="outputs")
    p.add_argument("--gpu", default="0", help="Physical CUDA device index")
    p.add_argument(
        "--llama", help="Compatibility alias for the first source model path"
    )
    p.add_argument(
        "--qwen", help="Compatibility alias for the second source model path"
    )
    return p


def selected_models(module, method, args):
    """Keep source checkpoints by default, or accept arbitrary replacement checkpoints."""
    config = getattr(module, "Config", getattr(module, "UserConfig", None))
    if method == "gpt_who":
        paths = list(module.MODEL_PATHS)
    elif hasattr(config, "PROXY_MODELS"):
        paths = [m["path"] for m in config.PROXY_MODELS]
    elif method == "profiler":
        paths = list(config.PROXY_MODEL_PATHS)
    elif method == "binoculars":
        paths = [config.OBSERVER_MODEL_PATH, config.PERFORMER_MODEL_PATH]
    else:
        paths = [config.MODEL_PATH]
    if args.models:
        if args.llama or args.qwen:
            raise ValueError("Use --models or the compatibility aliases, not both")
        paths = list(args.models)
    else:
        if args.llama:
            paths[0] = args.llama
        if args.qwen:
            if len(paths) < 2:
                raise ValueError("This method uses one scoring model")
            paths[1] = args.qwen
    if method == "binoculars" and len(paths) != 2:
        raise ValueError(
            "Binoculars requires two ordered models: observer and performer"
        )
    if method == "lrr" and len(paths) != 1:
        raise ValueError("LRR uses one scoring model")
    return paths


def configure(module, method, args):
    """Configure task paths and checkpoints without changing numerical settings."""
    import torch

    output = str(Path(args.output).resolve())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    name = args.task or Path(args.test).stem
    paths = selected_models(module, method, args)
    config = getattr(module, "Config", getattr(module, "UserConfig", None))
    if config is not None:
        config.DEVICE = device
        if hasattr(config, "GPU_ID"):
            config.GPU_ID = 0
        if hasattr(config, "GPU_INDEX"):
            config.GPU_INDEX = 0
        if hasattr(config, "OUTPUT_DIR"):
            config.OUTPUT_DIR = output
        if hasattr(config, "OUTPUT_BASE_DIR"):
            config.OUTPUT_BASE_DIR = output
        config.TRAIN_DATA_PATH = args.train
        config.TEST_DATA_PATH = args.test
        if hasattr(config, "TASK_NAMES"):
            config.TASK_NAMES = [name]
        if hasattr(config, "TASKS"):
            config.TASKS = [
                {
                    "task_name": name,
                    "train": args.train,
                    "test": args.test,
                    "output_dir": output,
                }
            ]
        if hasattr(config, "DATASETS"):
            limit = None
            config.DATASETS = [
                {"name": name, "test_path": args.test, "max_samples": limit}
            ]
        if hasattr(config, "PROXY_MODELS"):
            if args.models:
                config.PROXY_MODELS = [
                    {
                        "type": f"proxy_{i}_{hashlib.sha256(path.encode()).hexdigest()[:10]}",
                        "path": path,
                    }
                    for i, path in enumerate(paths)
                ]
            else:
                for model, path in zip(config.PROXY_MODELS, paths):
                    model["path"] = path
        if method == "profiler":
            config.PROXY_MODEL_PATHS = paths
        if method == "binoculars":
            config.OBSERVER_MODEL_PATH, config.PERFORMER_MODEL_PATH = paths
        if method == "lrr":
            config.MODEL_PATH = paths[0]
    if method == "gpt_who":
        module.DEVICE = device
        module.MODEL_PATHS = paths
        module.DATASETS_CONFIG = [
            {
                "task_name": name,
                "train_file": args.train,
                "test_file": args.test,
                "output_dir": output,
            }
        ]
    module._resolved_models = paths
    return module


def validate_labels(method, args):
    """Keep every class; report incompatible supervised label sets instead of filtering."""
    datasets = {}
    for split, path in [("train", args.train), ("test", args.test)]:
        if not path:
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not data:
            raise ValueError(f"{split}: expected a nonempty label-to-text-list object")
        normalized = [label.strip().lower() for label in data]
        if len(normalized) != len(set(normalized)):
            raise ValueError(
                f"{split}: labels collide after source whitespace/case normalization"
            )
        for label, texts in data.items():
            if label != label.strip():
                raise ValueError(
                    f"{split}: remove surrounding whitespace from label {label!r}"
                )
            if (
                not isinstance(texts, list)
                or not texts
                or not any(isinstance(t, str) and t.strip() for t in texts)
            ):
                raise ValueError(f"{split}: class {label!r} has no valid texts")
            if not all(isinstance(t, str) for t in texts):
                raise ValueError(
                    f"{split}: class {label!r} contains non-string samples"
                )
        datasets[split] = data
    if method in SUPERVISED:
        train_labels = set(datasets["train"])
        unknown = set(datasets["test"]) - train_labels
        if unknown:
            raise ValueError(
                f"Test classes absent from training: {sorted(unknown)}; no class was removed"
            )
    return {split: list(data) for split, data in datasets.items()}


def run_one(method, args):
    if method in SUPERVISED and not args.train:
        raise ValueError(f"--train is required for {method}")
    labels = validate_labels(method, args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("MPLBACKEND", "Agg")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    module = configure(importlib.import_module(METHODS[method]), method, args)
    metadata = {
        "method": method,
        "task": args.task or Path(args.test).stem,
        "train": args.train,
        "test": args.test,
        "models": module._resolved_models,
        "gpu": args.gpu,
        "classes": labels,
        "per_class_sample_limit": None,
        "data_sha256": {
            split: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for split, path in (("train", args.train), ("test", args.test)) if path
        },
    }
    metadata_path = output / "run_arguments.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        check_keys = ("method", "task", "train", "test", "models", "data_sha256")
        if any(previous.get(key) != metadata.get(key) for key in check_keys):
            raise ValueError(
                f"Output directory {output} contains a different experiment; "
                "choose a fresh --output directory to avoid stale feature caches."
            )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    module.main()


def run_single(module_name):
    method = next(k for k, v in METHODS.items() if v == module_name)
    run_one(method, parser(method).parse_args())
