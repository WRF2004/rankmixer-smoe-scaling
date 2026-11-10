"""Train and evaluate reproducible KuaiSAR ranking baselines.

Examples:
    python -m rmscaling.train --dataset synthetic --model rankmixer --epochs 1
    python -m rmscaling.train --dataset kuaisar --data-dir data/KuaiSAR --task rec
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import collate_batch, prepare_kuaisar
from .metrics import binary_auc, binary_logloss, ndcg_at_k
from .models import DCNv2Ranker, MLPRanker, RankMixerRanker
from .synthetic import prepare_synthetic


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=["synthetic", "kuaisar"], default="synthetic")
    result.add_argument("--data-dir", type=Path, default=Path("data/KuaiSAR"))
    result.add_argument("--task", choices=["rec", "search"], default="rec")
    result.add_argument("--model", choices=["mlp", "dcnv2", "rankmixer"], default="rankmixer")
    result.add_argument("--embedding-dim", type=int, default=32)
    result.add_argument("--hidden-dim", type=int, default=128)
    result.add_argument("--layers", type=int, default=2)
    result.add_argument("--ff-ratio", type=int, default=2)
    result.add_argument("--experts", type=int, default=1)
    result.add_argument("--top-k", type=int, default=1)
    result.add_argument("--shared-expert", action="store_true")
    result.add_argument("--balance-weight", type=float, default=0.01)
    result.add_argument("--batch-size", type=int, default=512)
    result.add_argument("--epochs", type=int, default=3)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--weight-decay", type=float, default=1e-6)
    result.add_argument("--patience", type=int, default=2)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    result.add_argument("--num-workers", type=int, default=0)
    result.add_argument("--max-rows", type=int, default=200_000, help="0 means full dataset")
    result.add_argument("--synthetic-rows", type=int, default=1024)
    result.add_argument("--min-freq", type=int, default=1)
    result.add_argument("--max-vocab", type=int, default=200_000)
    result.add_argument("--val-ratio", type=float, default=0.1)
    result.add_argument("--test-ratio", type=float, default=0.1)
    result.add_argument("--latency-batches", type=int, default=30)
    result.add_argument("--output-dir", type=Path, default=Path("results/smoke"))
    result.add_argument("--save-model", action="store_true")
    result.add_argument("--skip-test", action="store_true", help="keep held-out test labels sealed during sweeps")
    return result


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace, spec: Any) -> nn.Module:
    shared = dict(
        vocab_sizes=spec.vocab_sizes,
        num_dense=len(spec.numerical_fields),
        embedding_dim=args.embedding_dim,
        token_groups=spec.token_groups,
    )
    if args.model == "mlp":
        return MLPRanker(**shared, hidden_dims=(args.hidden_dim,) * args.layers)
    if args.model == "dcnv2":
        return DCNv2Ranker(**shared, hidden_dims=(args.hidden_dim,) * args.layers, cross_depth=args.layers)
    return RankMixerRanker(
        **shared,
        num_layers=args.layers,
        ffn_hidden_dim=args.embedding_dim * args.ff_ratio,
        num_experts=args.experts,
        top_k=args.top_k,
        shared_expert=args.shared_expert,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def model_output(model: nn.Module, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if hasattr(model, "forward_with_aux"):
        logits, aux, stats = model.forward_with_aux(batch)
        return logits, aux, stats
    logits = model(batch)
    return logits, logits.new_zeros(()), {}


def _stats_to_json(stats: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return convert(stats)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, task: str) -> dict[str, Any]:
    model.eval()
    labels: list[float] = []
    scores: list[float] = []
    groups: list[int] = []
    expert_counts: torch.Tensor | None = None
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            logits, _, stats = model_output(model, batch)
            labels.extend(batch["label"].detach().cpu().tolist())
            scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
            if "group_id" in batch:
                value = batch["group_id"]
                groups.extend(value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value)
            if "expert_counts" in stats:
                counts = torch.as_tensor(stats["expert_counts"], device="cpu", dtype=torch.float64)
                expert_counts = counts if expert_counts is None else expert_counts + counts
    result: dict[str, Any] = {
        "rows": len(labels),
        "positive_rate": sum(labels) / len(labels) if labels else None,
        "auc": binary_auc(labels, scores) if labels else None,
        "logloss": binary_logloss(labels, scores) if labels else None,
    }
    if task == "search" and groups:
        result["ndcg_at_10"], result["ndcg_groups"] = ndcg_at_k(labels, scores, groups)
    if expert_counts is not None:
        result["expert_counts"] = expert_counts.tolist()
        result["expert_utilization"] = (expert_counts / expert_counts.sum(dim=-1, keepdim=True).clamp_min(1)).tolist()
    return result


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    balance_weight: float,
) -> dict[str, float]:
    model.train()
    samples = 0
    total_task = 0.0
    total_aux = 0.0
    start = time.perf_counter()
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, aux, _ = model_output(model, batch)
        task_loss = criterion(logits, batch["label"].float())
        loss = task_loss + balance_weight * aux
        loss.backward()
        optimizer.step()
        size = int(batch["label"].shape[0])
        samples += size
        total_task += float(task_loss.detach()) * size
        total_aux += float(aux.detach()) * size
    elapsed = time.perf_counter() - start
    return {
        "task_loss": total_task / samples,
        "aux_loss": total_aux / samples,
        "seconds": elapsed,
        "rows_per_second": samples / elapsed,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_latency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    timed_batches: int,
) -> dict[str, float] | None:
    if timed_batches < 1:
        return None
    first = next(iter(loader), None)
    if first is None:
        return None
    batch = move_batch(first, device)
    model.eval()
    with torch.inference_mode():
        for _ in range(5):
            model(batch)
        _synchronize(device)
        times: list[float] = []
        for _ in range(timed_batches):
            _synchronize(device)
            started = time.perf_counter()
            model(batch)
            _synchronize(device)
            times.append((time.perf_counter() - started) * 1000.0)
    return {
        "batch_size": int(batch["label"].shape[0]),
        "median_ms": statistics.median(times),
        "p95_ms": sorted(times)[math.ceil(0.95 * len(times)) - 1],
        "batches": len(times),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs < 1 or args.batch_size < 1 or args.layers < 1:
        raise ValueError("epochs, batch_size and layers must be positive")
    if args.experts < 1 or not 1 <= args.top_k <= args.experts:
        raise ValueError("require 1 <= top_k <= experts")
    if args.model != "rankmixer" and args.experts != 1:
        raise ValueError("experts > 1 applies only to RankMixer")
    if args.shared_expert and (args.model != "rankmixer" or args.experts == 1):
        raise ValueError("shared-expert requires sparse RankMixer")
    if not 0 < args.val_ratio < 1 or not 0 < args.test_ratio < 1 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("invalid validation/test split")
    seed_all(args.seed)
    device = resolve_device(args.device)
    if args.dataset == "synthetic":
        data = prepare_synthetic(n_rows=args.synthetic_rows, seed=args.seed, task=args.task)
    else:
        data = prepare_kuaisar(
            args.data_dir,
            task=args.task,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            max_rows=None if args.max_rows == 0 else args.max_rows,
            min_freq=args.min_freq,
            max_vocab=args.max_vocab,
        )
    generator = torch.Generator().manual_seed(args.seed)
    common_loader = dict(batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_batch)
    train_loader = DataLoader(data.train, shuffle=True, generator=generator, **common_loader)
    val_loader = DataLoader(data.val, shuffle=False, **common_loader)
    test_loader = DataLoader(data.test, shuffle=False, **common_loader)
    model = build_model(args, data.spec).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(model, train_loader, optimizer, criterion, device, args.balance_weight)
        val_stats = evaluate(model, val_loader, device, args.task)
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        print(
            f"epoch={epoch} task_loss={train_stats['task_loss']:.5f} "
            f"val_logloss={val_stats['logloss']:.5f} val_auc={val_stats['auc']}",
            flush=True,
        )
        if val_stats["logloss"] < best_loss:
            best_loss = val_stats["logloss"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    train_seconds = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    val = evaluate(model, val_loader, device, args.task)
    test = None if args.skip_test else evaluate(model, test_loader, device, args.task)
    latency = measure_latency(model, val_loader, device, args.latency_batches)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    embedding_parameters = (
        model.embedding_parameter_count()
        if callable(getattr(model, "embedding_parameter_count", None))
        else int(getattr(model, "embedding_parameter_count", 0))
    )
    interaction_summary = model.parameter_summary() if hasattr(model, "parameter_summary") else {}
    non_embedding_parameters = total_parameters - embedding_parameters
    active_non_embedding = non_embedding_parameters
    if interaction_summary:
        active_non_embedding -= (
            interaction_summary["interaction_parameters"]
            - interaction_summary["active_interaction_parameters_per_example"]
        )
    result = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "dataset_metadata": data.metadata,
        "feature_spec": {
            "categorical_fields": list(data.spec.categorical_fields),
            "numerical_fields": list(data.spec.numerical_fields),
            "vocab_sizes": list(data.spec.vocab_sizes),
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "parameters": {
            "total": total_parameters,
            "embedding": embedding_parameters,
            "non_embedding": non_embedding_parameters,
            "active_non_embedding_per_example": active_non_embedding,
            **interaction_summary,
        },
        "best_epoch": min(history, key=lambda row: row["val"]["logloss"])["epoch"],
        "train_seconds": train_seconds,
        "history": history,
        "val": val,
        "test": test,
        "latency": latency,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    if args.save_model:
        torch.save({"state_dict": model.state_dict(), "config": result["config"]}, args.output_dir / "model.pt")
    print(json.dumps({"test": test, "latency": latency, "parameters": result["parameters"]}, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
