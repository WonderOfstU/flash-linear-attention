#!/usr/bin/env python3
"""Visualize retrieval heads from a model config.json.

This script only depends on Python standard library and one input file:
config.json containing a "headwise_config" field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _load_headwise_matrix(config_path: Path) -> Tuple[dict, List[int], List[List[int]]]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    raw = cfg.get("headwise_config")
    if raw is None:
        raise ValueError(f'"headwise_config" not found in {config_path}')
    if not isinstance(raw, dict):
        raise ValueError('"headwise_config" must be a dict')

    layer_keys: List[int] = []
    for key in raw:
        if str(key).isdigit():
            layer_keys.append(int(key))
    if not layer_keys:
        raise ValueError("No numeric layer keys found under headwise_config")

    layer_keys.sort()
    matrix: List[List[int]] = []
    expected_heads = None
    for layer_idx in layer_keys:
        values = raw[str(layer_idx)]
        if not isinstance(values, list):
            raise ValueError(f"Layer {layer_idx} headwise value must be a list")
        if expected_heads is None:
            expected_heads = len(values)
        if len(values) != expected_heads:
            raise ValueError(
                f"Inconsistent head count: layer {layer_idx} has {len(values)}, "
                f"expected {expected_heads}"
            )
        if any(v not in (0, 1) for v in values):
            raise ValueError(f"Layer {layer_idx} contains values not in {{0,1}}")
        matrix.append(values)

    return cfg, layer_keys, matrix


def _collect_retrieval_heads(
    layer_keys: List[int], matrix: List[List[int]]
) -> Tuple[Dict[int, List[int]], List[Tuple[int, int]]]:
    by_layer: Dict[int, List[int]] = {}
    pairs: List[Tuple[int, int]] = []
    for layer_idx, heads in zip(layer_keys, matrix):
        retrieval = [head_idx for head_idx, flag in enumerate(heads) if flag == 1]
        by_layer[layer_idx] = retrieval
        pairs.extend((layer_idx, head_idx) for head_idx in retrieval)
    return by_layer, pairs


def _write_svg_heatmap(
    layer_keys: List[int],
    matrix: List[List[int]],
    out_path: Path,
    title: str = "Headwise Retrieval Heatmap",
) -> None:
    n_layers = len(layer_keys)
    n_heads = len(matrix[0]) if matrix else 0

    cell = 14
    margin_left = 84
    margin_top = 56
    margin_right = 24
    margin_bottom = 48
    width = margin_left + n_heads * cell + margin_right
    height = margin_top + n_layers * cell + margin_bottom

    retrieval_color = "#d73027"
    non_retrieval_color = "#4575b4"
    grid_color = "#f2f2f2"
    text_color = "#222222"

    svg: List[str] = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    svg.append(
        f'<text x="{width // 2}" y="28" text-anchor="middle" font-size="16" '
        f'font-family="sans-serif" fill="{text_color}">{title}</text>'
    )

    for row, layer_idx in enumerate(layer_keys):
        y = margin_top + row * cell
        if row % 2 == 0:
            svg.append(
                f'<rect x="{margin_left}" y="{y}" width="{n_heads * cell}" height="{cell}" '
                f'fill="{grid_color}" opacity="0.18"/>'
            )
        for col, flag in enumerate(matrix[row]):
            x = margin_left + col * cell
            color = retrieval_color if flag == 1 else non_retrieval_color
            svg.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" '
                'stroke="white" stroke-width="0.5"/>'
            )

        label_y = y + cell * 0.75
        svg.append(
            f'<text x="{margin_left - 8}" y="{label_y}" text-anchor="end" '
            f'font-size="10" font-family="monospace" fill="{text_color}">L{layer_idx}</text>'
        )

    for col in range(n_heads):
        if col % 4 == 0 or col == n_heads - 1:
            x = margin_left + col * cell + cell / 2
            svg.append(
                f'<text x="{x}" y="{height - 20}" text-anchor="middle" font-size="9" '
                f'font-family="monospace" fill="{text_color}">H{col}</text>'
            )

    legend_x = margin_left
    legend_y = 36
    svg.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="12" height="12" fill="{retrieval_color}"/>'
    )
    svg.append(
        f'<text x="{legend_x + 18}" y="{legend_y + 10}" font-size="11" '
        'font-family="sans-serif" fill="#333333">retrieval (1)</text>'
    )
    svg.append(
        f'<rect x="{legend_x + 128}" y="{legend_y}" width="12" height="12" fill="{non_retrieval_color}"/>'
    )
    svg.append(
        f'<text x="{legend_x + 146}" y="{legend_y + 10}" font-size="11" '
        'font-family="sans-serif" fill="#333333">non-retrieval (0)</text>'
    )

    svg.append("</svg>")
    out_path.write_text("\n".join(svg), encoding="utf-8")


def _write_outputs(
    config_path: Path,
    out_dir: Path,
    layer_keys: List[int],
    matrix: List[List[int]],
    retrieval_by_layer: Dict[int, List[int]],
    retrieval_pairs: List[Tuple[int, int]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    total_layers = len(layer_keys)
    total_heads_per_layer = len(matrix[0])
    total_heads = total_layers * total_heads_per_layer
    retrieval_total = len(retrieval_pairs)
    retrieval_ratio = retrieval_total / total_heads if total_heads else 0.0

    json_payload = {
        "config_path": str(config_path),
        "num_layers": total_layers,
        "num_heads_per_layer": total_heads_per_layer,
        "total_heads": total_heads,
        "retrieval_heads_total": retrieval_total,
        "retrieval_ratio": retrieval_ratio,
        "retrieval_heads_by_layer": {
            str(k): v for k, v in retrieval_by_layer.items()
        },
        "all_retrieval_heads": [
            {"layer": layer_idx, "head": head_idx}
            for layer_idx, head_idx in retrieval_pairs
        ],
    }
    (out_dir / "retrieval_heads.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines: List[str] = []
    lines.append(f"config_path: {config_path}")
    lines.append(
        "summary: "
        f"layers={total_layers}, heads_per_layer={total_heads_per_layer}, "
        f"retrieval={retrieval_total}/{total_heads} ({retrieval_ratio:.2%})"
    )
    lines.append("")
    lines.append("retrieval_heads_by_layer:")
    for layer_idx in layer_keys:
        heads = retrieval_by_layer[layer_idx]
        lines.append(f"  L{layer_idx:02d}: {heads}")
    lines.append("")
    lines.append("all_retrieval_heads (layer, head):")
    lines.extend(f"  ({layer_idx}, {head_idx})" for layer_idx, head_idx in retrieval_pairs)
    (out_dir / "retrieval_heads.txt").write_text("\n".join(lines), encoding="utf-8")

    _write_svg_heatmap(layer_keys, matrix, out_dir / "headwise_heatmap.svg")

    print(lines[1])
    print("per-layer retrieval heads:")
    for layer_idx in layer_keys:
        print(f"  L{layer_idx:02d}: {retrieval_by_layer[layer_idx]}")
    print("")
    print("all retrieval heads (layer, head):")
    for layer_idx, head_idx in retrieval_pairs:
        print(f"  ({layer_idx}, {head_idx})")
    print("")
    print(f"Saved: {out_dir / 'headwise_heatmap.svg'}")
    print(f"Saved: {out_dir / 'retrieval_heads.txt'}")
    print(f"Saved: {out_dir / 'retrieval_heads.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate retrieval-head heatmap and full list from a model config.json "
            "that contains headwise_config."
        )
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to config.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./headwise_report"),
        help="Output directory (default: ./headwise_report)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, layer_keys, matrix = _load_headwise_matrix(args.config)
    _ = cfg  # keep for future extension
    retrieval_by_layer, retrieval_pairs = _collect_retrieval_heads(layer_keys, matrix)
    _write_outputs(
        config_path=args.config.resolve(),
        out_dir=args.out_dir.resolve(),
        layer_keys=layer_keys,
        matrix=matrix,
        retrieval_by_layer=retrieval_by_layer,
        retrieval_pairs=retrieval_pairs,
    )


if __name__ == "__main__":
    main()
