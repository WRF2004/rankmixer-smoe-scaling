"""Small deterministic KuaiSAR-shaped CSV fixtures for smoke experiments.

Synthetic results are useful for checking that the pipeline runs. They are not
evidence of a ranking improvement on the real KuaiSAR dataset.
"""

from __future__ import annotations

import csv
import math
import random
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .data import PreparedData, prepare_kuaisar


_CHINA = timezone(timedelta(hours=8))
_START = datetime(2023, 5, 22, 14, 30, tzinfo=_CHINA)


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_synthetic_kuaisar(
    data_dir: str | Path,
    *,
    n_users: int = 32,
    n_items: int = 128,
    n_rows: int = 512,
    seed: int = 42,
) -> Path:
    """Write rec, search, user, and item CSVs using official KuaiSAR columns."""

    if n_users < 2 or n_items < 4 or n_rows < 12:
        raise ValueError("synthetic data requires >=2 users, >=4 items, and >=12 rec rows")
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    user_rows: list[dict[str, object]] = []
    for user_id in range(1, n_users + 1):
        user_rows.append({
            "user_id": user_id,
            "search_active_level": user_id % 5,
            "rec_active_level": (user_id + 1) % 5,
            "onehot_feat1": user_id % 3,
            "onehot_feat2": user_id % 8,
        })
    _write_csv(
        directory / "user_features.csv",
        ("user_id", "search_active_level", "rec_active_level", "onehot_feat1", "onehot_feat2"),
        user_rows,
    )

    item_rows: list[dict[str, object]] = []
    duration_by_item: dict[int, int] = {}
    category_by_item: dict[int, int] = {}
    for item_id in range(1, n_items + 1):
        category = item_id % 8
        duration_by_item[item_id] = 10_000 + 1_000 * (item_id % 90)
        category_by_item[item_id] = category
        item_rows.append({
            "item_id": item_id,
            "caption": str([category, item_id % 17]),
            "author_id": 1 + item_id % max(2, n_items // 8),
            "item_type": "NORMAL" if item_id % 7 else "AD",
            "upload_time": (_START - timedelta(days=1 + item_id % 45)).strftime("%Y-%m-%d"),
            "upload_type": "LongImport" if item_id % 2 else "Original",
            "first_level_category_id": category,
            "first_level_category_name": f"category_{category}",
            "first_level_category_name_en": f"category_{category}",
            "second_level_category_id": item_id % 24,
            "second_level_category_name": f"subcategory_{item_id % 24}",
            "second_level_category_name_en": f"subcategory_{item_id % 24}",
            "third_level_category_id": item_id % 5,
            "third_level_category_name": f"level_3_{item_id % 5}",
            "third_level_category_name_en": f"level_3_{item_id % 5}",
            "fourth_level_category_id": item_id % 3,
            "fourth_level_category_name": f"level_4_{item_id % 3}",
            "fourth_level_category_name_en": f"level_4_{item_id % 3}",
        })
    _write_csv(
        directory / "item_features.csv",
        (
            "item_id", "caption", "author_id", "item_type", "upload_time",
            "upload_type", "first_level_category_id", "first_level_category_name",
            "first_level_category_name_en", "second_level_category_id",
            "second_level_category_name", "second_level_category_name_en",
            "third_level_category_id", "third_level_category_name",
            "third_level_category_name_en", "fourth_level_category_id",
            "fourth_level_category_name", "fourth_level_category_name_en",
        ),
        item_rows,
    )

    def sampled_click(user_id: int, item_id: int) -> int:
        category_match = int(user_id % 8 == category_by_item[item_id])
        score = -1.1 + 2.0 * category_match + 0.6 * (item_id % 5 == user_id % 5)
        probability = 1 / (1 + math.exp(-score))
        return int(rng.random() < probability)

    rec_rows: list[dict[str, object]] = []
    for index in range(n_rows):
        user_id = rng.randint(1, n_users)
        item_id = rng.randint(1, n_items)
        moment = _START + timedelta(seconds=index * 60)
        click = sampled_click(user_id, item_id)
        rec_rows.append({
            "user_id": user_id,
            "item_id": item_id,
            # This is post-exposure feedback and must not become a model input.
            "playing_time": 30_000 if click else 300,
            "duration_ms": duration_by_item[item_id],
            "time": moment.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": int(moment.timestamp() * 1000),
            "click": click,
            "forward": click if index % 10 == 0 else 0,
            "like": click if index % 3 == 0 else 0,
            "follow": click if index % 13 == 0 else 0,
            "search": click if index % 7 == 0 else 0,
            "search_item_related": click if index % 7 == 0 else 0,
        })
    _write_csv(
        directory / "rec_inter.csv",
        (
            "user_id", "item_id", "playing_time", "duration_ms", "time",
            "timestamp", "click", "forward", "like", "follow", "search",
            "search_item_related",
        ),
        rec_rows,
    )

    search_rows: list[dict[str, object]] = []
    n_sessions = max(6, n_rows // 8)
    for session_id in range(1, n_sessions + 1):
        user_id = rng.randint(1, n_users)
        moment = _START + timedelta(seconds=session_id * 180)
        for position in range(4):
            item_id = rng.randint(1, n_items)
            search_rows.append({
                "user_id": user_id,
                "search_session_id": session_id,
                "search_session_time": moment.strftime("%Y-%m-%d %H:%M:%S"),
                "search_session_timestamp": int(moment.timestamp() * 1000),
                "keyword": str([user_id % 8, session_id % 17]),
                "search_source": "USER_INPUT" if session_id % 2 else "RECOMMENDED_QUERY",
                "item_id": item_id,
                "click_cnt": sampled_click(user_id, item_id),
                "item_type": "VIDEO" if position < 3 else "IMAGE_ATLAS",
            })
    _write_csv(
        directory / "src_inter.csv",
        (
            "user_id", "search_session_id", "search_session_time",
            "search_session_timestamp", "keyword", "search_source", "item_id",
            "click_cnt", "item_type",
        ),
        search_rows,
    )
    return directory


def prepare_synthetic(
    *,
    n_rows: int = 512,
    seed: int = 42,
    task: str = "rec",
) -> PreparedData:
    """Prepare a tiny, deterministic dataset without a network download."""

    with tempfile.TemporaryDirectory(prefix="rmscaling-synthetic-") as directory:
        write_synthetic_kuaisar(directory, n_rows=n_rows, seed=seed)
        prepared = prepare_kuaisar(directory, task=task, max_vocab=1_000)
    return prepared
