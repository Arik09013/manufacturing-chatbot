"""
Matplotlib comparison figures for the benchmark (matplotlib only, no seaborn).

generate_all(agg, per_backend, figdir) writes:
    quality_bar.png        composite quality per backend
    radar.png              multi-metric radar (quality/faithfulness/readability/…)
    latency_bar.png        mean latency per backend
    hallucination_bar.png  mean hallucination rate (grounded routes)
    faithfulness_bar.png   mean numeric faithfulness (grounded routes)
    quality_by_route.png   grouped bars: quality per route per backend
    cost_vs_quality.png    scatter of estimated cost vs quality

All figures are self-contained PNGs suitable for a thesis appendix.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

# Colour-blind-safe categorical palette (Okabe–Ito), assigned per backend.
_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
            "#56B4E9", "#F0E442", "#000000"]


def _colors(n: int) -> list[str]:
    return [_PALETTE[i % len(_PALETTE)] for i in range(n)]


def _label(s: dict) -> str:
    return f"{s['backend']}\n{s['model']}"


def _bar(ax, labels, values, title, ylabel, colors, ylim=None, fmt="{:.2f}"):
    values = [0 if v is None else v for v in values]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def generate_all(agg: list[dict], per_backend: dict, figdir: Path) -> None:
    if not agg:
        return
    figdir.mkdir(parents=True, exist_ok=True)
    labels = [_label(s) for s in agg]
    colors = _colors(len(agg))

    # 1) Composite quality.
    fig, ax = plt.subplots(figsize=(8, 5))
    _bar(ax, labels, [s["composite_quality"] for s in agg],
         "LLM response quality (composite, 0–1)", "composite quality", colors, (0, 1))
    _save(fig, figdir / "quality_bar.png")

    # 2) Latency.
    fig, ax = plt.subplots(figsize=(8, 5))
    _bar(ax, labels, [s["latency_s"] for s in agg],
         "Mean latency per response", "seconds", colors, fmt="{:.2f}")
    _save(fig, figdir / "latency_bar.png")

    # 3) Hallucination (grounded routes).
    fig, ax = plt.subplots(figsize=(8, 5))
    _bar(ax, labels, [s["hallucination_rate"] for s in agg],
         "Mean hallucination rate (grounded routes, lower = better)",
         "hallucination rate", colors, (0, 1), fmt="{:.3f}")
    _save(fig, figdir / "hallucination_bar.png")

    # 4) Faithfulness (grounded routes).
    fig, ax = plt.subplots(figsize=(8, 5))
    _bar(ax, labels, [s["faithfulness"] for s in agg],
         "Mean numeric faithfulness (grounded routes)", "faithfulness", colors, (0, 1),
         fmt="{:.3f}")
    _save(fig, figdir / "faithfulness_bar.png")

    # 5) Radar (multi-metric fingerprint).
    _radar(agg, figdir / "radar.png")

    # 6) Quality by route (grouped bars).
    _quality_by_route(per_backend, figdir / "quality_by_route.png")

    # 7) Cost vs quality scatter.
    _cost_vs_quality(agg, figdir / "cost_vs_quality.png", colors)


def _radar(agg: list[dict], path: Path) -> None:
    import math
    axes_metrics = [
        ("composite_quality", "quality"),
        ("faithfulness", "faithfulness"),
        ("readability_operator", "readability"),
        ("conciseness_score", "conciseness"),
        ("safety_score", "safety"),
        ("determinism_score", "determinism"),
    ]
    labels = [m[1] for m in axes_metrics]
    n = len(labels)
    angles = [i / n * 2 * math.pi for i in range(n)] + [0.0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colors = _colors(len(agg))
    for s, c in zip(agg, colors):
        vals = [(s.get(k) or 0.0) for k, _ in axes_metrics]
        vals += vals[:1]
        ax.plot(angles, vals, color=c, linewidth=2, label=f"{s['backend']}")
        ax.fill(angles, vals, color=c, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Model quality fingerprint (normalized 0–1)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
    _save(fig, path)


def _quality_by_route(per_backend: dict, path: Path) -> None:
    import numpy as np
    backends = [b for b, rows in per_backend.items() if rows]
    if not backends:
        return
    routes = sorted({r["route"] for rows in per_backend.values() for r in rows})
    x = np.arange(len(routes))
    width = 0.8 / max(1, len(backends))
    colors = _colors(len(backends))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, backend in enumerate(backends):
        rows = per_backend[backend]
        means = []
        for route in routes:
            vals = [r["composite_quality"] for r in rows if r["route"] == route]
            means.append(sum(vals) / len(vals) if vals else 0)
        ax.bar(x + i * width, means, width, label=backend, color=colors[i],
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x + width * (len(backends) - 1) / 2)
    ax.set_xticklabels(routes)
    ax.set_ylabel("composite quality")
    ax.set_ylim(0, 1)
    ax.set_title("Response quality by route and backend", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, path)


def _cost_vs_quality(agg: list[dict], path: Path, colors: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for s, c in zip(agg, colors):
        cost = (s.get("est_cost_usd") or 0) * 1000  # $ per 1000 responses
        ax.scatter(cost, s["composite_quality"], s=140, color=c,
                   edgecolor="black", zorder=3)
        ax.annotate(s["backend"], (cost, s["composite_quality"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("estimated cost, USD per 1000 responses")
    ax.set_ylabel("composite quality")
    ax.set_ylim(0, 1)
    ax.set_title("Cost vs quality (local models sit at $0)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    _save(fig, path)
