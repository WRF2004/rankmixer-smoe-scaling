"""Leak-aware, dependency-light preparation for KuaiSAR ranking experiments.

The public dataset contains recommendation interactions in ``rec_inter.csv`` and
search results in ``src_inter.csv``. Only information available before the
interaction is used as a model feature. Interactions are split by timestamp,
and categorical vocabularies are fitted on the training period alone.
"""

from __future__ import annotations

import csv
import math
from array import array
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Mapping, Sequence


_REC_FIELDS = (
    "user_id", "user_onehot_feat1", "user_onehot_feat2", "item_id",
    "author_id", "item_type", "upload_type", "category_1", "category_2",
    "category_3", "category_4", "hour_of_day", "day_of_week",
)
_REC_GROUPS = (
    "user", "user", "user", "item", "item", "item", "item", "item",
    "item", "item", "item", "context", "context",
)
_SEARCH_FIELDS = _REC_FIELDS + ("query", "search_source", "result_type")
_SEARCH_GROUPS = _REC_GROUPS + ("context", "context", "item")
_ITEM_FIELDS = (
    "author_id", "item_type", "upload_type", "first_level_category_id",
    "second_level_category_id", "third_level_category_id",
    "fourth_level_category_id", "upload_time",
)
_EMPTY_ITEM = ("",) * len(_ITEM_FIELDS)
_EMPTY_USER = ("", "")
_TZ_CHINA = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class FeatureSpec:
    """Field-level categorical IDs, with 0=OOV and 1=missing for every field."""

    categorical_fields: tuple[str, ...]
    vocab_sizes: tuple[int, ...]
    numerical_fields: tuple[str, ...]
    field_groups: tuple[str, ...]
    task: str
    vocabularies: Mapping[str, Mapping[str, int]]

    @property
    def token_groups(self) -> dict[str, tuple[int, ...]]:
        groups: dict[str, list[int]] = {}
        for index, group in enumerate(self.field_groups):
            groups.setdefault(group, []).append(index)
        return {name: tuple(indices) for name, indices in groups.items()}


class PackedDataset:
    """Compact columnar in-memory data, compatible with a Torch DataLoader."""

    def __init__(self, n_categorical: int, n_numerical: int, *, search: bool):
        self.categorical_columns = [array("I") for _ in range(n_categorical)]
        self.numerical_columns = [array("f") for _ in range(n_numerical)]
        self.labels = array("b")
        self.timestamps = array("Q")
        self.group_ids = array("I") if search else None

    def append(
        self,
        categorical: Sequence[int],
        numerical: Sequence[float],
        label: int,
        timestamp: int,
        group_id: int | None = None,
    ) -> None:
        if len(categorical) != len(self.categorical_columns):
            raise ValueError("categorical feature count does not match schema")
        if len(numerical) != len(self.numerical_columns):
            raise ValueError("numerical feature count does not match schema")
        for column, value in zip(self.categorical_columns, categorical):
            column.append(value)
        for column, value in zip(self.numerical_columns, numerical):
            column.append(value)
        self.labels.append(int(label))
        self.timestamps.append(int(timestamp))
        if self.group_ids is not None:
            if group_id is None:
                raise ValueError("search rows require a group_id")
            self.group_ids.append(group_id)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        result: dict[str, object] = {
            "categorical": tuple(column[index] for column in self.categorical_columns),
            "numerical": tuple(column[index] for column in self.numerical_columns),
            "label": float(self.labels[index]),
            "timestamp": int(self.timestamps[index]),
        }
        if self.group_ids is not None:
            result["group_id"] = int(self.group_ids[index])
        return result


@dataclass(frozen=True)
class PreparedData:
    train: PackedDataset
    val: PackedDataset
    test: PackedDataset
    spec: FeatureSpec
    metadata: Mapping[str, object]


def collate_batch(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Convert Python dataset rows to tensors; Torch is imported only here."""

    if not rows:
        raise ValueError("cannot collate an empty batch")
    import torch

    result: dict[str, object] = {
        "categorical": torch.tensor([r["categorical"] for r in rows], dtype=torch.long),
        "numerical": torch.tensor([r["numerical"] for r in rows], dtype=torch.float32),
        "label": torch.tensor([r["label"] for r in rows], dtype=torch.float32),
        "timestamp": torch.tensor([r["timestamp"] for r in rows], dtype=torch.long),
    }
    if "group_id" in rows[0]:
        result["group_id"] = torch.tensor([r["group_id"] for r in rows], dtype=torch.long)
    return result


def _resolve_file(data_dir: str | Path, task: str) -> tuple[Path, Path]:
    path = Path(data_dir).expanduser()
    names = ("rec_inter.csv",) if task == "rec" else ("src_inter.csv", "search_inter.csv")
    if path.is_file():
        if path.name not in names:
            raise ValueError(f"expected one of {names}, got {path.name}")
        return path, path.parent
    if not path.is_dir():
        raise FileNotFoundError(f"KuaiSAR directory does not exist: {path}")
    matches = sorted(p for name in names for p in path.rglob(name) if p.is_file())
    if not matches:
        raise FileNotFoundError(f"none of {names} found under {path}")
    if len(matches) != 1:
        raise ValueError("multiple interaction files found; pass the desired CSV path explicitly")
    return matches[0], matches[0].parent


def _read_interactions(path: Path, max_rows: int | None) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"empty interaction CSV: {path}")
        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break
            yield row


def _timestamp(row: Mapping[str, str], task: str) -> int:
    field = "timestamp" if task == "rec" else "search_session_timestamp"
    raw = row.get(field, "")
    if not raw:
        raise ValueError(f"missing {field} in interaction row")
    try:
        value = int(float(raw))
    except ValueError as exc:
        raise ValueError(f"invalid {field}: {raw!r}") from exc
    if value < 0:
        raise ValueError(f"negative {field}: {value}")
    return value


def _require_columns(path: Path, task: str) -> None:
    required = {"user_id", "item_id", "timestamp", "click"} if task == "rec" else {
        "user_id", "item_id", "search_session_id", "search_session_timestamp",
        "keyword", "search_source", "click_cnt", "item_type",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        fields = csv.DictReader(stream).fieldnames or []
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")


def _split_boundaries(timestamps: array, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be positive and sum to less than 1")
    if len(timestamps) < 3:
        raise ValueError("at least three interaction rows are required")
    ordered = sorted(timestamps)
    n = len(ordered)
    unique = 1 + sum(ordered[index] != ordered[index - 1] for index in range(1, n))
    if unique < 3:
        raise ValueError("at least three distinct timestamps are needed for three chronological splits")
    first_target = max(1, min(n - 2, int(n * (1 - val_ratio - test_ratio))))
    second_target = max(first_target + 1, min(n - 1, int(n * (1 - test_ratio))))
    train_index = bisect_left(ordered, ordered[first_target])
    if train_index == 0:
        train_index = bisect_right(ordered, ordered[0])
    test_index = bisect_left(ordered, ordered[second_target])
    if test_index <= train_index:
        test_index = bisect_right(ordered, ordered[train_index])
    if test_index == n:
        test_index = train_index
        train_index = bisect_left(ordered, ordered[train_index - 1])
    if not (0 < train_index < test_index < n):
        raise ValueError("cannot form nonempty chronological train/val/test splits")
    return ordered[train_index], ordered[test_index]


def _side_rows(path: Path, ids: set[str], fields: Sequence[str]) -> dict[str, tuple[str, ...]]:
    if not path.exists():
        return {}
    result: dict[str, tuple[str, ...]] = {}
    id_field = "user_id" if path.name == "user_features.csv" else "item_id"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            key = row.get(id_field, "")
            if key in ids:
                result[key] = tuple((row.get(field) or "") for field in fields)
    return result


@lru_cache(maxsize=131_072)
def _upload_ms(raw: str) -> int | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.strip())
        if (parsed.year, parsed.month, parsed.day) == (1970, 1, 1):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_TZ_CHINA)
        return int(parsed.timestamp() * 1000)
    except ValueError:
        return None


def _safe_duration(raw: str) -> tuple[float, float]:
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return 0.0, 1.0
    if not math.isfinite(number) or number < 0:
        return 0.0, 1.0
    return math.log1p(number), 0.0


def _features(
    row: Mapping[str, str],
    task: str,
    user_features: Mapping[str, tuple[str, str]],
    item_features: Mapping[str, tuple[str, ...]],
    timestamp: int,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    user = user_features.get(row["user_id"], _EMPTY_USER)
    item = item_features.get(row["item_id"], _EMPTY_ITEM)
    local_seconds = timestamp // 1000 + 8 * 3600
    hour = str((local_seconds // 3600) % 24)
    weekday = str((local_seconds // 86400 + 3) % 7)  # Unix epoch was Thursday.
    categorical = (
        row["user_id"], user[0], user[1], row["item_id"], item[0],
        item[1], item[2], item[3], item[4], item[5], item[6], hour, weekday,
    )
    upload_ms = _upload_ms(item[7])
    if upload_ms is None or upload_ms > timestamp:
        age = (0.0, 1.0)
    else:
        age = (math.log1p((timestamp - upload_ms) / 86_400_000), 0.0)
    if task == "rec":
        duration = _safe_duration(row.get("duration_ms", ""))
        numerical = (duration[0], age[0], duration[1], age[1])
    else:
        categorical += (
            "".join((row.get("keyword") or "").split()),
            row.get("search_source") or "",
            row.get("item_type") or "",
        )
        numerical = age
    return categorical, numerical


def _vocab_limit(field: str, max_vocab: int | Mapping[str, int] | None) -> int | None:
    if isinstance(max_vocab, Mapping):
        limit = max_vocab.get(field, 200_000)
    else:
        limit = max_vocab
    if limit is not None and limit < 1:
        raise ValueError("max_vocab must allow at least one learned category")
    return limit


def prepare_kuaisar(
    data_dir: str | Path,
    *,
    task: str = "rec",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_rows: int | None = None,
    min_freq: int = 1,
    max_vocab: int | Mapping[str, int] | None = 200_000,
) -> PreparedData:
    """Prepare a chronological recommendation CTR or search click experiment.

    ``max_rows`` selects the first N rows in the interaction CSV. It is useful
    for development, but full-data results should omit it. Exact timestamp ties
    stay in one split, preserving search sessions. Labels and post-exposure
    feedback are never included in the feature vectors.
    """

    if task not in {"rec", "search"}:
        raise ValueError("task must be 'rec' or 'search'")
    if max_rows is not None and max_rows < 3:
        raise ValueError("max_rows must be at least 3")
    if min_freq < 1:
        raise ValueError("min_freq must be at least 1")
    interaction_path, directory = _resolve_file(data_dir, task)
    _require_columns(interaction_path, task)
    timestamps = array("Q")
    user_ids: set[str] = set()
    item_ids: set[str] = set()
    session_timestamps: dict[tuple[str, str], int] = {}
    for row in _read_interactions(interaction_path, max_rows):
        timestamp = _timestamp(row, task)
        timestamps.append(timestamp)
        user_ids.add(row["user_id"])
        item_ids.add(row["item_id"])
        if task == "search":
            key = (row["user_id"], row["search_session_id"])
            prior = session_timestamps.setdefault(key, timestamp)
            if prior != timestamp:
                raise ValueError(f"search session {key} has inconsistent timestamps")
    del session_timestamps
    train_cut, test_cut = _split_boundaries(timestamps, val_ratio, test_ratio)
    del timestamps

    users = _side_rows(directory / "user_features.csv", user_ids, ("onehot_feat1", "onehot_feat2"))
    items = _side_rows(directory / "item_features.csv", item_ids, _ITEM_FIELDS)
    fields = _REC_FIELDS if task == "rec" else _SEARCH_FIELDS
    groups = _REC_GROUPS if task == "rec" else _SEARCH_GROUPS
    numerical_fields = (
        ("log1p_duration_ms", "log1p_item_age_days", "duration_missing", "upload_age_missing")
        if task == "rec" else ("log1p_item_age_days", "upload_age_missing")
    )
    counts = {field: Counter() for field in fields}
    for row in _read_interactions(interaction_path, max_rows):
        timestamp = _timestamp(row, task)
        if timestamp >= train_cut:
            continue
        categorical, _ = _features(row, task, users, items, timestamp)
        for field, value in zip(fields, categorical):
            if value:
                counts[field][value] += 1

    vocabularies: dict[str, dict[str, int]] = {}
    for field in fields:
        values = ((value, count) for value, count in counts[field].items() if count >= min_freq)
        ordered = sorted(values, key=lambda pair: (-pair[1], pair[0]))
        limit = _vocab_limit(field, max_vocab)
        if limit is not None:
            ordered = ordered[:limit]
        vocabularies[field] = {value: index + 2 for index, (value, _) in enumerate(ordered)}
    del counts

    spec = FeatureSpec(
        categorical_fields=fields,
        vocab_sizes=tuple(len(vocabularies[field]) + 2 for field in fields),
        numerical_fields=numerical_fields,
        field_groups=groups,
        task=task,
        vocabularies=vocabularies,
    )
    datasets = {
        name: PackedDataset(len(fields), len(numerical_fields), search=task == "search")
        for name in ("train", "val", "test")
    }
    group_lookup: dict[tuple[str, str], int] = {}
    for row in _read_interactions(interaction_path, max_rows):
        timestamp = _timestamp(row, task)
        split = "train" if timestamp < train_cut else "val" if timestamp < test_cut else "test"
        categorical, numerical = _features(row, task, users, items, timestamp)
        encoded = tuple(
            1 if not value else vocabularies[field].get(value, 0)
            for field, value in zip(fields, categorical)
        )
        if task == "rec":
            label = 1 if float(row["click"]) > 0 else 0
            group_id = None
        else:
            label = 1 if float(row["click_cnt"]) > 0 else 0
            key = (row["user_id"], row["search_session_id"])
            group_id = group_lookup.setdefault(key, len(group_lookup))
        datasets[split].append(encoded, numerical, label, timestamp, group_id)
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("chronological split produced an empty partition; use more rows")
    metadata = {
        "source": str(interaction_path),
        "task": task,
        "split_counts": {name: len(data) for name, data in datasets.items()},
        "train_before_ms": train_cut,
        "test_from_ms": test_cut,
        "selected_rows": sum(len(data) for data in datasets.values()),
        "max_rows": max_rows,
        "min_freq": min_freq,
    }
    return PreparedData(
        train=datasets["train"], val=datasets["val"], test=datasets["test"],
        spec=spec, metadata=metadata,
    )
