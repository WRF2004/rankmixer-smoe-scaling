"""Render measured sweep results as compact SVG figures.

No chart is emitted when the input contains no successful measured runs.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any


COLORS = {"mlp": "#68707d", "dcnv2": "#2573b6", "rankmixer": "#e06431"}


def _scatter(rows: list[dict[str, str]], x_field: str, title: str, x_label: str, output: Path, log_x: bool = False) -> None:
    points: list[tuple[float, float, dict[str, str]]] = []
    for row in rows:
        try:
            x_value = float(row[x_field])
            y_value = float(row["val_auc"])
            if not math.isfinite(x_value) or not math.isfinite(y_value) or x_value <= 0:
                continue
            points.append((math.log10(x_value) if log_x else x_value, y_value, row))
        except (ValueError, KeyError):
            continue
    if not points:
        return
    width, height = 760, 480
    left, right, top, bottom = 92, 28, 65, 76
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xpad = (xmax - xmin) * 0.08 or max(abs(xmin) * 0.05, 0.5)
    ypad = (ymax - ymin) * 0.12 or 0.005
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = max(0.0, ymin - ypad), min(1.0, ymax + ypad)
    plot_width, plot_height = width - left - right, height - top - bottom

    def px(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_width

    def py(value: float) -> float:
        return top + plot_height - (value - ymin) / (ymax - ymin) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="Arial,sans-serif" font-size="20" font-weight="bold" fill="#20242a">{html.escape(title)}</text>',
    ]
    for step in range(6):
        value = ymin + (ymax - ymin) * step / 5
        y = py(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e8eaed"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#555">{value:.4f}</text>')
    for step in range(6):
        value = xmin + (xmax - xmin) * step / 5
        x = px(value)
        label = f"{10**value:.0f}" if log_x else f"{value:.2f}"
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#555">{label}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#555"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#555"/>')
    for x_value, y_value, row in points:
        model = row["model"]
        color = COLORS.get(model, "#e06431")
        sparse = model == "rankmixer" and int(row.get("experts", "1")) > 1
        marker = 'stroke="#20242a" stroke-width="1.8"' if sparse else 'stroke="#ffffff" stroke-width="1.5"'
        label = f"{row['name']}: AUC={y_value:.5f}"
        parts.append(
            f'<circle cx="{px(x_value):.1f}" cy="{py(y_value):.1f}" r="6.5" fill="{color}" {marker}>'
            f"<title>{html.escape(label)}</title></circle>"
        )
    parts.append(f'<text x="{left+plot_width/2:.1f}" y="{height-16}" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#333">{html.escape(x_label)}</text>')
    parts.append(f'<text transform="translate(23,{top+plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#333">Validation AUC</text>')
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def _heatmap(metrics: dict[str, Any], output: Path) -> None:
    held_out = metrics.get("test") or {}
    validation = metrics.get("val") or {}
    utilization = validation.get("expert_utilization") or held_out.get("expert_utilization")
    if not utilization:
        return
    if utilization and isinstance(utilization[0][0], list):
        matrix = utilization[0]
    else:
        matrix = utilization
    if not matrix or not matrix[0]:
        return
    tokens, experts = len(matrix), len(matrix[0])
    cell = 34
    left, top = 90, 70
    width, height = left + experts * cell + 24, top + tokens * cell + 45
    max_value = max(float(value) for row in matrix for value in row) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="30" font-family="Arial,sans-serif" font-size="19" font-weight="bold">Expert utilization (layer 1)</text>',
    ]
    for expert in range(experts):
        parts.append(f'<text x="{left+expert*cell+cell/2:.0f}" y="{top-10}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11">E{expert}</text>')
    for token, row in enumerate(matrix):
        parts.append(f'<text x="{left-9}" y="{top+token*cell+22}" text-anchor="end" font-family="Arial,sans-serif" font-size="11">T{token}</text>')
        for expert, value in enumerate(row):
            shade = int(245 - 185 * float(value) / max_value)
            parts.append(
                f'<rect x="{left+expert*cell}" y="{top+token*cell}" width="{cell-2}" height="{cell-2}" fill="rgb({shade},{shade+5},{min(255,shade+10)})">'
                f'<title>token {token}, expert {expert}: {float(value):.3f}</title></rect>'
            )
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--routing-metrics", type=Path)
    args = parser.parse_args()
    with args.summary.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "ok"]
    if not rows:
        raise SystemExit("No successful measured runs in summary")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _scatter(rows, "median_latency_ms", "Validation AUC vs. measured latency", "Median batch latency (ms)", args.output_dir / "val_auc_vs_latency.svg")
    _scatter(rows, "non_embedding_params", "Validation AUC vs. interaction parameters", "Non-embedding parameters (log scale)", args.output_dir / "val_auc_vs_parameters.svg", log_x=True)
    _scatter(rows, "active_non_embedding_params", "Validation AUC vs. active parameters", "Active non-embedding parameters (log scale)", args.output_dir / "val_auc_vs_active_parameters.svg", log_x=True)
    if args.routing_metrics:
        with args.routing_metrics.open(encoding="utf-8") as handle:
            _heatmap(json.load(handle), args.output_dir / "expert_utilization.svg")
    print(args.output_dir)


if __name__ == "__main__":
    main()
