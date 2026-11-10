import json

from rmscaling.train import parser, run


def test_sparse_training_keeps_test_sealed_and_records_budget(tmp_path):
    args = parser().parse_args(
        [
            "--dataset", "synthetic",
            "--task", "search",
            "--synthetic-rows", "64",
            "--model", "rankmixer",
            "--experts", "2",
            "--epochs", "1",
            "--batch-size", "16",
            "--latency-batches", "1",
            "--device", "cpu",
            "--skip-test",
            "--output-dir", str(tmp_path),
        ]
    )
    result = run(args)
    assert result["test"] is None
    assert result["val"]["rows"] > 0
    assert result["parameters"]["active_non_embedding_per_example"] < result["parameters"]["non_embedding"]
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["test"] is None
    assert saved["config"]["skip_test"] is True
