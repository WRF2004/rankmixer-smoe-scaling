"""Run a controlled width/expert scaling sweep and collect measured results."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _ints(value: str) -> list[int]:
    numbers = [int(part) for part in value.split(",") if part.strip()]
    if not numbers or any(number < 1 for number in numbers):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return numbers


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=["synthetic", "kuaisar"], default="kuaisar")
    result.add_argument("--data-dir", type=Path, default=Path("data/KuaiSAR"))
    result.add_argument("--task", choices=["rec", "search"], default="rec")
    result.add_argument("--widths", type=_ints, default=[32, 64])
    result.add_argument("--expert-counts", type=_ints, default=[2, 4, 8])
    result.add_argument("--seeds", type=_ints, default=[42])
    result.add_argument("--devices", default="auto", help="comma-separated devices, e.g. cuda:0,cuda:1")
    result.add_argument("--epochs", type=int, default=3)
    result.add_argument("--batch-size", type=int, default=512)
    result.add_argument("--max-rows", type=int, default=200_000)
    result.add_argument("--synthetic-rows", type=int, default=1024)
    result.add_argument("--hidden-dim", type=int, default=128)
    result.add_argument("--layers", type=int, default=2)
    result.add_argument("--ff-ratio", type=int, default=2)
    result.add_argument("--top-k", type=int, default=1)
    result.add_argument("--output-dir", type=Path, default=Path("results/sweep"))
    result.add_argument("--dry-run", action="store_true")
    return result


def configurations(args: argparse.Namespace) -> list[dict[str, int | str]]:
    items: list[dict[str, int | str]] = []
    for seed, width in itertools.product(args.seeds, args.widths):
        for model in ("mlp", "dcnv2", "rankmixer"):
            items.append({"model": model, "width": width, "experts": 1, "seed": seed})
        for experts in args.expert_counts:
            if args.top_k > experts:
                raise ValueError("top-k cannot exceed any expert count")
            items.append({"model": "rankmixer", "width": width, "experts": experts, "seed": seed})
    return items


def _run_one(args: argparse.Namespace, config: dict[str, int | str], device: str) -> dict[str, Any]:
    name = f"{config['model']}_d{config['width']}_e{config['experts']}_s{config['seed']}"
    output = args.output_dir / name
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "rmscaling.train",
        "--dataset", args.dataset,
        "--data-dir", str(args.data_dir),
        "--task", args.task,
        "--model", str(config["model"]),
        "--embedding-dim", str(config["width"]),
        "--hidden-dim", str(args.hidden_dim),
        "--layers", str(args.layers),
        "--ff-ratio", str(args.ff_ratio),
        "--experts", str(config["experts"]),
        "--top-k", str(args.top_k),
        "--seed", str(config["seed"]),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--max-rows", str(args.max_rows),
        "--synthetic-rows", str(args.synthetic_rows),
        "--device", device,
        "--output-dir", str(output),
        "--skip-test",
    ]
    with (output / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        return {"name": name, "status": "failed", "log": str(output / "run.log")}
    with (output / "metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    val = metrics["val"]
    return {
        "name": name,
        "status": "ok",
        "model": config["model"],
        "width": config["width"],
        "experts": config["experts"],
        "top_k": args.top_k,
        "seed": config["seed"],
        "dataset": args.dataset,
        "task": args.task,
        "rows": metrics["dataset_metadata"].get("selected_rows", ""),
        "val_auc": val["auc"],
        "val_logloss": val["logloss"],
        "val_ndcg_at_10": val.get("ndcg_at_10"),
        "total_params": metrics["parameters"]["total"],
        "non_embedding_params": metrics["parameters"]["non_embedding"],
        "active_non_embedding_params": metrics["parameters"]["active_non_embedding_per_example"],
        "median_latency_ms": metrics["latency"]["median_ms"] if metrics["latency"] else None,
        "train_seconds": metrics["train_seconds"],
    }


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    devices = [part.strip() for part in args.devices.split(",") if part.strip()]
    if not devices:
        raise ValueError("at least one device is required")
    configs = configurations(args)
    if args.dry_run:
        for index, config in enumerate(configs):
            print({**config, "device": devices[index % len(devices)]})
        return []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queues = [configs[index::len(devices)] for index in range(len(devices))]

    def worker(device: str, assigned: list[dict[str, int | str]]) -> list[dict[str, Any]]:
        return [_run_one(args, config, device) for config in assigned]

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, device, assigned) for device, assigned in zip(devices, queues)]
        results = [row for future in futures for row in future.result()]
    columns = [
        "name", "status", "model", "width", "experts", "top_k", "seed", "dataset", "task",
        "rows", "val_auc", "val_logloss", "val_ndcg_at_10", "total_params",
        "non_embedding_params", "active_non_embedding_params", "median_latency_ms", "train_seconds", "log",
    ]
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    failures = sum(row["status"] != "ok" for row in results)
    print(f"completed={len(results) - failures} failed={failures} summary={args.output_dir / 'summary.csv'}")
    return results


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
