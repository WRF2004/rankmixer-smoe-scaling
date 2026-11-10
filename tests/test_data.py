"""Data safety and reproducibility checks; executable with stdlib unittest."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from array import array
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rmscaling.data import _split_boundaries, prepare_kuaisar  # noqa: E402
from rmscaling.synthetic import prepare_synthetic, write_synthetic_kuaisar  # noqa: E402


class DataPipelineTests(unittest.TestCase):
    def test_rec_is_chronological_and_deterministic(self) -> None:
        first = prepare_synthetic(n_rows=96, seed=7)
        second = prepare_synthetic(n_rows=96, seed=7)
        self.assertEqual(sum(first.metadata["split_counts"].values()), 96)
        self.assertLess(max(first.train.timestamps), min(first.val.timestamps))
        self.assertLess(max(first.val.timestamps), min(first.test.timestamps))
        self.assertEqual(first.spec.vocabularies, second.spec.vocabularies)
        self.assertEqual(first.train[0], second.train[0])
        self.assertIn("user", first.spec.token_groups)
        self.assertIn("item", first.spec.token_groups)
        self.assertIn("context", first.spec.token_groups)
        self.assertEqual(len(first.train[0]["categorical"]), len(first.spec.vocab_sizes))
        self.assertEqual(len(first.train[0]["numerical"]), len(first.spec.numerical_fields))

    def test_rec_feedback_is_not_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write_synthetic_kuaisar(directory, n_rows=48, seed=5)
            before = prepare_kuaisar(directory)
            path = Path(directory) / "rec_inter.csv"
            with path.open("r", newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            for row in rows:
                row["playing_time"] = "999999"
                row["forward"] = "1"
                row["like"] = "1"
                row["follow"] = "1"
                row["search"] = "1"
                row["search_item_related"] = "1"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            after = prepare_kuaisar(directory)
            self.assertEqual(before.train[0]["categorical"], after.train[0]["categorical"])
            self.assertEqual(before.train[0]["numerical"], after.train[0]["numerical"])
            forbidden = {"playing_time", "forward", "like", "follow", "search", "search_item_related"}
            self.assertTrue(forbidden.isdisjoint(after.spec.categorical_fields))
            self.assertTrue(forbidden.isdisjoint(after.spec.numerical_fields))

    def test_validation_and_test_unknown_values_are_oov(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write_synthetic_kuaisar(directory, n_rows=60, seed=11)
            path = Path(directory) / "rec_inter.csv"
            with path.open("r", newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            rows[-1]["user_id"] = "999999"
            rows[-1]["item_id"] = "888888"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            prepared = prepare_kuaisar(directory)
            self.assertNotIn("999999", prepared.spec.vocabularies["user_id"])
            self.assertNotIn("888888", prepared.spec.vocabularies["item_id"])
            self.assertEqual(prepared.test[-1]["categorical"][0], 0)
            self.assertEqual(prepared.test[-1]["categorical"][3], 0)

    def test_search_sessions_stay_in_one_split(self) -> None:
        prepared = prepare_synthetic(n_rows=128, seed=2, task="search")
        self.assertLess(max(prepared.train.timestamps), min(prepared.val.timestamps))
        self.assertLess(max(prepared.val.timestamps), min(prepared.test.timestamps))
        groups = [set(dataset.group_ids) for dataset in (prepared.train, prepared.val, prepared.test)]
        self.assertTrue(groups[0].isdisjoint(groups[1]))
        self.assertTrue(groups[0].isdisjoint(groups[2]))
        self.assertTrue(groups[1].isdisjoint(groups[2]))
        self.assertIn("group_id", prepared.train[0])
        self.assertNotIn("search_session_id", prepared.spec.categorical_fields)
        self.assertNotIn("click_cnt", prepared.spec.categorical_fields)

    def test_invalid_task_and_short_fixture(self) -> None:
        with self.assertRaisesRegex(ValueError, "task"):
            prepare_synthetic(task="unknown")
        with self.assertRaisesRegex(ValueError, "rec rows"):
            write_synthetic_kuaisar("/tmp/unused-fixture", n_rows=3)

    def test_large_timestamp_ties_keep_three_nonempty_periods(self) -> None:
        self.assertEqual(
            _split_boundaries(array("Q", [0] * 5 + [1] * 5 + [2] * 90), 0.1, 0.1),
            (1, 2),
        )


if __name__ == "__main__":
    unittest.main()
