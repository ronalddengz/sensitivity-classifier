"""
C-Score Necessity Calibration Benchmark
========================================

Runs all three scorers (E-Score, Q-Score, C-Score) on example inputs,
compares their outputs, and generates plots + analysis to determine
when the expensive C-Score (LLM) can safely be skipped.

Usage:
    python benchmark.py
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from example_inputs import load_example_inputs

# ---------------------------------------------------------------------------
# Scorer imports  (modules use hyphens so we use importlib)
# ---------------------------------------------------------------------------
from importlib import import_module


def _import_e_scorer():
    """Import and return the E-Score detector class."""
    mod = import_module("e-score")
    return mod.PIIDetector


def _import_q_scorer():
    """Import and return the Q-Score analyzer class."""
    mod = import_module("q-score")
    return mod.QScoreAnalyzer


def _import_c_scorer():
    """Import and return the C-Score detector class."""
    mod = import_module("c-score")
    return mod.NarrativeSensitivityDetector


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

C_SCORE_THRESHOLD = 0.2   # risk_score >= this → C-Score "found something"
FALSE_NEG_TOLERANCE = 0.05  # max acceptable miss rate when choosing skip threshold
OUTPUT_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Running the scorers
# ---------------------------------------------------------------------------

def run_e_score(examples: list[dict]) -> list[dict]:
    """Run the E-Score detector on all examples."""
    print("=" * 60)
    print("Running E-Score (Presidio PII/PHI detection)...")
    print("=" * 60)

    PIIDetector = _import_e_scorer()
    detector = PIIDetector()

    results = []
    for i, ex in enumerate(examples, 1):
        name = ex.get("name", f"Example {i}")
        result = detector.analyze(ex["text"])
        results.append({
            "name": name,
            "e_score": result.e_score,
            "entity_count": len(result.entities),
            "entities": [
                {
                    "type": e.entity_type,
                    "text": e.text,
                    "confidence": e.confidence,
                    "weight": e.weight,
                    "weighted_score": e.weighted_score,
                }
                for e in result.entities
            ],
        })
        print(f"  [{i}/{len(examples)}] {name}: e_score={result.e_score:.4f}  "
              f"entities={len(result.entities)}")

    print()
    return results


def run_q_score(examples: list[dict]) -> list[dict]:
    """Run the Q-Score analyzer on all examples."""
    print("=" * 60)
    print("Running Q-Score (k-anonymity quasi-identifier risk)...")
    print("=" * 60)

    QScoreAnalyzer = _import_q_scorer()
    analyzer = QScoreAnalyzer()

    results = []
    for i, ex in enumerate(examples, 1):
        name = ex.get("name", f"Example {i}")
        q_result = analyzer.analyze(ex["text"])
        results.append({
            "name": name,
            "q_score": q_result.q_score,
            "expected_k": q_result.expected_k,
            "num_qis": len(q_result.detected_qis),
        })
        print(f"  [{i}/{len(examples)}] {name}: q_score={q_result.q_score:.4f}  "
              f"E[k]={q_result.expected_k:.2f}  QIs={len(q_result.detected_qis)}")

    print()
    return results


def run_c_score(examples: list[dict]) -> list[dict]:
    """Run the C-Score detector on all examples."""
    print("=" * 60)
    print("Running C-Score (LLM contextual narrative analysis)...")
    print("This requires Ollama running locally with llama3.2:3b")
    print("=" * 60)

    NarrativeSensitivityDetector = _import_c_scorer()
    detector = NarrativeSensitivityDetector()

    results = []
    for i, ex in enumerate(examples, 1):
        name = ex.get("name", f"Example {i}")
        try:
            analysis = detector.analyze(ex["text"])
            factor_details = {}
            for f in analysis.factors:
                factor_details[f.name] = {
                    "detected": f.detected,
                    "confidence": f.confidence,
                    "explanation": f.explanation,
                }

            results.append({
                "name": name,
                "risk_score": analysis.risk_score,
                "risk_level": analysis.overall_risk.value,
                "factors": factor_details,
                "num_factors_detected": sum(
                    1 for f in analysis.factors if f.detected
                ),
                "error": None,
            })
            print(f"  [{i}/{len(examples)}] {name}: risk_score={analysis.risk_score:.4f}  "
                  f"level={analysis.overall_risk.value}  "
                  f"factors={sum(1 for f in analysis.factors if f.detected)}/7")

        except Exception as e:
            print(f"  [{i}/{len(examples)}] {name}: ERROR — {e}")
            results.append({
                "name": name,
                "risk_score": None,
                "risk_level": None,
                "factors": {},
                "num_factors_detected": 0,
                "error": str(e),
            })

    print()
    return results


# ---------------------------------------------------------------------------
# Combine and derive metrics
# ---------------------------------------------------------------------------

def combine_results(
    examples: list[dict],
    e_results: list[dict],
    q_results: list[dict],
    c_results: list[dict],
) -> list[dict]:
    """Merge scorer outputs and compute derived metrics."""

    combined = []
    for ex, e, q, c in zip(examples, e_results, q_results, c_results):
        e_score = e["e_score"]
        q_score = q["q_score"]
        c_risk = c["risk_score"]  # may be None on error

        # Derived metrics
        combined_eq = max(e_score, q_score)

        c_flagged = (c_risk is not None and c_risk >= C_SCORE_THRESHOLD)
        c_additive = (
            c_flagged and combined_eq < 0.5
        )  # C-Score found something that E+Q missed

        record = {
            "name": ex.get("name", ""),
            "text_preview": ex["text"][:120] + "..." if len(ex["text"]) > 120 else ex["text"],
            "expected_critical": ex.get("expected_critical", False),

            # Raw scores
            "e_score": round(e_score, 4),
            "q_score": round(q_score, 4),
            "c_risk_score": round(c_risk, 4) if c_risk is not None else None,
            "c_risk_level": c["risk_level"],

            # Derived
            "combined_eq_score": round(combined_eq, 4),
            "c_score_flagged": c_flagged,
            "c_score_additive": c_additive,

            # Detail counts
            "e_entity_count": e["entity_count"],
            "q_num_qis": q["num_qis"],
            "q_expected_k": round(q["expected_k"], 4),
            "c_num_factors": c["num_factors_detected"],

            # C-Score factor detail
            "c_factors": c["factors"],
            "c_error": c["error"],
        }
        combined.append(record)

    return combined


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Shared palette
_RISK_COLORS = {
    "LOW": "#22c55e",
    "MEDIUM": "#eab308",
    "HIGH": "#f97316",
    "CRITICAL": "#ef4444",
    None: "#94a3b8",  # grey for errors
}


def _setup_plot_style():
    """Apply a consistent, clean plot style."""
    plt.rcParams.update({
        "figure.facecolor": "#0f172a",
        "axes.facecolor": "#1e293b",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#e2e8f0",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": "#334155",
        "grid.alpha": 0.5,
        "text.color": "#e2e8f0",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "legend.facecolor": "#1e293b",
        "legend.edgecolor": "#475569",
        "legend.fontsize": 9,
        "font.family": "sans-serif",
    })


def plot_eq_scatter(combined: list[dict], out_dir: Path):
    """Scatter plot: E-Score vs Q-Score, colored by C-Score risk level."""
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 7))

    for rec in combined:
        color = _RISK_COLORS.get(rec["c_risk_level"], "#94a3b8")
        marker = "D" if rec["c_score_additive"] else "o"
        edge = "#ffffff" if rec["c_score_additive"] else color
        ax.scatter(
            rec["e_score"], rec["q_score"],
            c=color, edgecolors=edge, linewidths=1.5,
            s=120, marker=marker, zorder=3,
        )

    # Labels for each point
    for rec in combined:
        short = rec["name"][:25]
        ax.annotate(
            short,
            (rec["e_score"], rec["q_score"]),
            textcoords="offset points", xytext=(6, 6),
            fontsize=7, color="#cbd5e1", alpha=0.85,
        )

    # Legend for risk levels
    for level, color in _RISK_COLORS.items():
        if level is None:
            continue
        ax.scatter([], [], c=color, s=60, label=f"C-Score: {level}")
    ax.scatter([], [], c="#94a3b8", edgecolors="#ffffff", linewidths=1.5,
               s=60, marker="D", label="C-Score additive (E+Q missed)")

    ax.set_xlabel("E-Score (Explicit PII/PHI)")
    ax.set_ylabel("Q-Score (Quasi-Identifier Risk)")
    ax.set_title("E-Score vs Q-Score — colored by C-Score Risk Level")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    path = out_dir / "scatter_eq_vs_c.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_combined_vs_c(combined: list[dict], out_dir: Path):
    """Scatter plot: max(E,Q) vs C-Score risk_score with decision boundary."""
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 7))

    xs, ys, colors = [], [], []
    for rec in combined:
        if rec["c_risk_score"] is None:
            continue
        xs.append(rec["combined_eq_score"])
        ys.append(rec["c_risk_score"])
        colors.append(_RISK_COLORS.get(rec["c_risk_level"], "#94a3b8"))

    ax.scatter(xs, ys, c=colors, s=120, edgecolors="#475569", linewidths=1, zorder=3)

    # Label points
    for rec in combined:
        if rec["c_risk_score"] is None:
            continue
        short = rec["name"][:25]
        ax.annotate(
            short,
            (rec["combined_eq_score"], rec["c_risk_score"]),
            textcoords="offset points", xytext=(6, 6),
            fontsize=7, color="#cbd5e1", alpha=0.85,
        )

    # Decision boundary lines
    ax.axhline(y=C_SCORE_THRESHOLD, color="#facc15", linestyle="--",
               alpha=0.7, label=f"C-Score threshold ({C_SCORE_THRESHOLD})")
    ax.axvline(x=0.5, color="#38bdf8", linestyle="--",
               alpha=0.7, label="E+Q threshold (0.5)")

    # Shade the "safe to skip C-Score" region
    ax.fill_between(
        [0.5, 1.05], -0.05, 1.05,
        alpha=0.08, color="#22c55e",
        label="Safe to skip C-Score",
    )
    # Shade the "C-Score needed" region
    ax.fill_between(
        [-0.05, 0.5], C_SCORE_THRESHOLD, 1.05,
        alpha=0.08, color="#ef4444",
        label="C-Score likely needed",
    )

    ax.set_xlabel("Combined E+Q Score — max(E-Score, Q-Score)")
    ax.set_ylabel("C-Score Risk Score")
    ax.set_title("When Does the C-Score Add Value?")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    path = out_dir / "scatter_combined_vs_c.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_heatmap(combined: list[dict], out_dir: Path):
    """Heatmap: E-Score (x) vs Q-Score (y) grid, cell color = C-Score flagged rate."""
    _setup_plot_style()

    # Bin edges
    n_bins = 5
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_labels_e = [f"{bin_edges[i]:.1f}–{bin_edges[i+1]:.1f}" for i in range(n_bins)]
    bin_labels_q = [f"{bin_edges[i]:.1f}–{bin_edges[i+1]:.1f}" for i in range(n_bins)]

    # Accumulate counts
    flagged_grid = np.zeros((n_bins, n_bins))
    count_grid = np.zeros((n_bins, n_bins))

    for rec in combined:
        if rec["c_risk_score"] is None:
            continue
        ei = min(int(rec["e_score"] * n_bins), n_bins - 1)
        qi = min(int(rec["q_score"] * n_bins), n_bins - 1)
        count_grid[qi, ei] += 1
        if rec["c_score_flagged"]:
            flagged_grid[qi, ei] += 1

    # Rate (avoid div-by-zero)
    with np.errstate(invalid="ignore"):
        rate_grid = np.where(count_grid > 0, flagged_grid / count_grid, np.nan)

    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = plt.cm.RdYlGn_r  # red = high flagged rate, green = low
    cmap.set_bad(color="#1e293b")  # empty cells

    im = ax.imshow(
        rate_grid, origin="lower", aspect="auto",
        cmap=cmap, vmin=0, vmax=1,
    )

    # Annotate cells with count and rate
    for qi in range(n_bins):
        for ei in range(n_bins):
            count = int(count_grid[qi, ei])
            if count > 0:
                rate = rate_grid[qi, ei]
                ax.text(
                    ei, qi,
                    f"{rate:.0%}\n(n={count})",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if rate > 0.5 else "#e2e8f0",
                )
            else:
                ax.text(ei, qi, "—", ha="center", va="center",
                        fontsize=9, color="#64748b")

    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_labels_e, fontsize=8)
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels(bin_labels_q, fontsize=8)
    ax.set_xlabel("E-Score Bin")
    ax.set_ylabel("Q-Score Bin")
    ax.set_title("C-Score Flagged Rate by E-Score × Q-Score Region")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("C-Score Flagged Rate", color="#e2e8f0")
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")

    fig.tight_layout()
    path = out_dir / "heatmap_eq_c_flagged.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_example_bars(combined: list[dict], out_dir: Path):
    """Grouped bar chart: E, Q, C scores side-by-side for each example."""
    _setup_plot_style()

    names = [r["name"][:30] for r in combined]
    e_scores = [r["e_score"] for r in combined]
    q_scores = [r["q_score"] for r in combined]
    c_scores = [r["c_risk_score"] if r["c_risk_score"] is not None else 0 for r in combined]

    n = len(names)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 7))

    bars_e = ax.bar(x - width, e_scores, width, label="E-Score",
                    color="#3b82f6", edgecolor="#1e40af", linewidth=0.5)
    bars_q = ax.bar(x, q_scores, width, label="Q-Score",
                    color="#8b5cf6", edgecolor="#5b21b6", linewidth=0.5)
    bars_c = ax.bar(x + width, c_scores, width, label="C-Score",
                    color="#ef4444", edgecolor="#991b1b", linewidth=0.5)

    # C-Score threshold line
    ax.axhline(y=C_SCORE_THRESHOLD, color="#facc15", linestyle="--",
               alpha=0.6, label=f"C-Score threshold ({C_SCORE_THRESHOLD})")

    ax.set_xlabel("Example Input")
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Per-Example Score Comparison: E-Score / Q-Score / C-Score")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    path = out_dir / "bars_per_example.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_threshold_sweep(combined: list[dict], out_dir: Path):
    """Sweep max(E,Q) thresholds and plot miss-rate vs skip-rate curves.

    At each candidate threshold *t*, examples with max(E,Q) >= t would be
    considered "safe" and C-Score would be skipped.  We measure:
      * skip_rate  – fraction of examples we'd skip  (efficiency)
      * miss_rate  – of the skipped examples, fraction where C-Score was
                     actually flagged  (danger / false-negative rate)
    """
    _setup_plot_style()

    valid = [r for r in combined if r["c_risk_score"] is not None]
    if len(valid) < 2:
        print("  ⚠ Not enough valid C-Score results for threshold sweep.")
        return

    thresholds = np.linspace(0.0, 1.0, 201)
    skip_rates = []
    miss_rates = []

    for t in thresholds:
        skipped = [r for r in valid if r["combined_eq_score"] >= t]
        n_skip = len(skipped)
        skip_rate = n_skip / len(valid)

        if n_skip == 0:
            miss_rate = 0.0
        else:
            missed = sum(1 for r in skipped if r["c_score_flagged"])
            miss_rate = missed / n_skip

        skip_rates.append(skip_rate)
        miss_rates.append(miss_rate)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_skip = "#38bdf8"
    color_miss = "#ef4444"

    ax1.plot(thresholds, skip_rates, color=color_skip, linewidth=2,
             label="Skip rate (efficiency)")
    ax1.set_xlabel("max(E, Q) skip threshold")
    ax1.set_ylabel("Skip rate", color=color_skip)
    ax1.tick_params(axis="y", labelcolor=color_skip)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.02, 1.02)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, miss_rates, color=color_miss, linewidth=2,
             linestyle="--", label="Miss rate (false negatives)")
    ax2.set_ylabel("Miss rate (of skipped)", color=color_miss)
    ax2.tick_params(axis="y", labelcolor=color_miss)
    ax2.set_ylim(-0.02, 1.02)

    # Shade the tolerance band
    ax2.axhline(y=FALSE_NEG_TOLERANCE, color="#facc15", linestyle=":",
                alpha=0.7, label=f"Miss-rate tolerance ({FALSE_NEG_TOLERANCE:.0%})")

    # Find optimal threshold (lowest t where miss_rate <= tolerance)
    optimal_t = None
    for t, mr in zip(thresholds, miss_rates):
        if mr <= FALSE_NEG_TOLERANCE:
            optimal_t = t
            break
    if optimal_t is not None:
        ax1.axvline(x=optimal_t, color="#22c55e", linestyle="-.",
                    alpha=0.8, linewidth=1.5)
        ax1.annotate(
            f"Optimal threshold ≈ {optimal_t:.2f}",
            xy=(optimal_t, 0.5), xytext=(optimal_t + 0.08, 0.65),
            fontsize=10, color="#22c55e",
            arrowprops=dict(arrowstyle="->", color="#22c55e"),
        )

    # Merged legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right",
              framealpha=0.9)

    ax1.set_title("Threshold Sweep: Skip Rate vs Miss Rate")
    fig.tight_layout()
    path = out_dir / "threshold_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_eq_c_correlation(combined: list[dict], out_dir: Path):
    """Scatter of max(E,Q) vs C-Score risk_score with linear regression."""
    _setup_plot_style()

    xs, ys = [], []
    for r in combined:
        if r["c_risk_score"] is not None:
            xs.append(r["combined_eq_score"])
            ys.append(r["c_risk_score"])

    if len(xs) < 3:
        print("  ⚠ Not enough data points for correlation plot.")
        return

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)

    # Linear regression via numpy
    coeffs = np.polyfit(xs_arr, ys_arr, 1)
    slope, intercept = coeffs
    fit_line = np.poly1d(coeffs)

    # Pearson r
    r_val = np.corrcoef(xs_arr, ys_arr)[0, 1]
    r_sq = r_val ** 2

    fig, ax = plt.subplots(figsize=(9, 7))

    colors = [_RISK_COLORS.get(r_rec["c_risk_level"], "#94a3b8")
              for r_rec in combined if r_rec["c_risk_score"] is not None]
    ax.scatter(xs, ys, c=colors, s=100, edgecolors="#475569",
              linewidths=0.8, zorder=3, alpha=0.85)

    # Regression line
    x_fit = np.linspace(0, 1, 100)
    ax.plot(x_fit, fit_line(x_fit), color="#facc15", linewidth=2,
            linestyle="--", label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")

    ax.set_xlabel("max(E-Score, Q-Score)")
    ax.set_ylabel("C-Score Risk Score")
    ax.set_title("Correlation: E+Q Combined vs C-Score")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Stats box
    stats_text = f"r = {r_val:.3f}\nR² = {r_sq:.3f}\nn = {len(xs)}"
    ax.text(0.97, 0.03, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b",
                      edgecolor="#475569", alpha=0.9),
            color="#e2e8f0", family="monospace")

    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    path = out_dir / "correlation_eq_c.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    return r_val, r_sq  # for the summary


def plot_agreement_matrix(combined: list[dict], eq_threshold: float,
                          out_dir: Path):
    """2×2 confusion matrix: E+Q prediction vs C-Score ground truth."""
    _setup_plot_style()

    valid = [r for r in combined if r["c_risk_score"] is not None]
    if not valid:
        print("  ⚠ No valid C-Score results for agreement matrix.")
        return

    # Confusion counts
    tp = fp = fn = tn = 0
    for r in valid:
        eq_pos = r["combined_eq_score"] >= eq_threshold
        c_pos = r["c_score_flagged"]
        if eq_pos and c_pos:
            tp += 1
        elif eq_pos and not c_pos:
            fp += 1
        elif not eq_pos and c_pos:
            fn += 1
        else:
            tn += 1

    matrix = np.array([[tn, fp], [fn, tp]])
    labels_pred = ["E+Q < thresh\n(would run C)", "E+Q ≥ thresh\n(would skip C)"]
    labels_truth = ["C-Score: NOT flagged", "C-Score: FLAGGED"]

    fig, ax = plt.subplots(figsize=(7, 6))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom", ["#1e293b", "#3b82f6", "#ef4444"], N=256)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    # Annotate cells
    total = len(valid)
    for i in range(2):
        for j in range(2):
            count = matrix[i, j]
            pct = count / total * 100 if total else 0
            color = "white" if count > 0 else "#64748b"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels_pred, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels_truth, fontsize=9)
    ax.set_xlabel("E+Q Decision (predicted)")
    ax.set_ylabel("C-Score (ground truth)")
    ax.set_title(f"Agreement Matrix @ max(E,Q) threshold = {eq_threshold:.2f}")

    # Precision / Recall / F1 footer
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else float("nan"))
    fnr = fn / (fn + tp) if (fn + tp) else 0.0  # false-negative rate

    footer = (f"Precision={precision:.2f}  Recall={recall:.2f}  "
              f"F1={f1:.2f}  FNR(miss)={fnr:.2%}")
    fig.text(0.5, 0.01, footer, ha="center", fontsize=10, color="#cbd5e1")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = out_dir / "agreement_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": f1, "fnr": fnr}


# ---------------------------------------------------------------------------
# Summary analysis
# ---------------------------------------------------------------------------

def _find_optimal_threshold(combined: list[dict],
                            tolerance: float = FALSE_NEG_TOLERANCE
                            ) -> tuple[float | None, float, float]:
    """Find the lowest max(E,Q) threshold where miss-rate ≤ *tolerance*.

    Returns (optimal_threshold, skip_rate_at_optimal, miss_rate_at_optimal).
    If no threshold satisfies the tolerance, returns (None, 0, 0).
    """
    valid = [r for r in combined if r["c_risk_score"] is not None]
    if not valid:
        return None, 0.0, 0.0

    thresholds = np.linspace(0.0, 1.0, 201)
    for t in thresholds:
        skipped = [r for r in valid if r["combined_eq_score"] >= t]
        if not skipped:
            continue
        n_missed = sum(1 for r in skipped if r["c_score_flagged"])
        mr = n_missed / len(skipped)
        if mr <= tolerance:
            sr = len(skipped) / len(valid)
            return float(round(t, 3)), sr, mr

    return None, 0.0, 0.0


def print_summary(combined: list[dict],
                  corr_stats: tuple | None = None,
                  agreement: dict | None = None):
    """Print the calibration summary and suggested thresholds."""

    valid = [r for r in combined if r["c_risk_score"] is not None]
    n = len(valid)

    if n == 0:
        print("\n⚠ No valid C-Score results — cannot produce summary.")
        return

    n_flagged = sum(1 for r in valid if r["c_score_flagged"])
    n_additive = sum(1 for r in valid if r["c_score_additive"])
    n_redundant = n_flagged - n_additive

    print()
    print("=" * 65)
    print("  CALIBRATION SUMMARY")
    print("=" * 65)
    print()
    print(f"  Total examples analysed:            {n}")
    print(f"  C-Score flagged (risk ≥ {C_SCORE_THRESHOLD}):       {n_flagged}/{n}")
    print(f"  C-Score additive (E+Q missed):       {n_additive}/{n}")
    print(f"  C-Score redundant (E+Q sufficient):  {n_redundant}/{n}")
    print(f"  C-Score not needed:                  {n - n_flagged}/{n}")
    print()

    # Break down by E+Q region
    eq_high = [r for r in valid if r["combined_eq_score"] >= 0.5]
    eq_mid  = [r for r in valid if 0.2 <= r["combined_eq_score"] < 0.5]
    eq_low  = [r for r in valid if r["combined_eq_score"] < 0.2]

    def _region_stats(label, subset):
        if not subset:
            print(f"  {label:35s}  (no examples)")
            return
        flagged = sum(1 for r in subset if r["c_score_flagged"])
        additive = sum(1 for r in subset if r["c_score_additive"])
        print(f"  {label:35s}  n={len(subset):2d}  "
              f"flagged={flagged}  additive={additive}")

    print("  By E+Q region:")
    _region_stats("HIGH  (max(E,Q) ≥ 0.5)", eq_high)
    _region_stats("MID   (0.2 ≤ max(E,Q) < 0.5)", eq_mid)
    _region_stats("LOW   (max(E,Q) < 0.2)", eq_low)

    # ── Correlation statistics ─────────────────────────────────────────
    if corr_stats is not None:
        r_val, r_sq = corr_stats
        print()
        print("-" * 65)
        print("  E+Q ↔ C-SCORE CORRELATION")
        print("-" * 65)
        print(f"  Pearson r:  {r_val:+.3f}")
        print(f"  R²:         {r_sq:.3f}")
        if r_sq >= 0.5:
            print("  → Strong correlation: E+Q is a good predictor of C-Score.")
        elif r_sq >= 0.25:
            print("  → Moderate correlation: E+Q captures some C-Score signal.")
        else:
            print("  → Weak correlation: C-Score catches fundamentally")
            print("    different things than E+Q — be cautious skipping it.")

    # ── Optimal threshold via sweep ────────────────────────────────────
    print()
    print("-" * 65)
    print("  OPTIMAL SKIP THRESHOLD  "
          f"(≤ {FALSE_NEG_TOLERANCE:.0%} false-negative tolerance)")
    print("-" * 65)

    opt_t, opt_sr, opt_mr = _find_optimal_threshold(combined)

    if opt_t is not None:
        print(f"  → Recommended threshold: max(E,Q) ≥ {opt_t:.2f}")
        print(f"     Skip rate:  {opt_sr:.1%} of inputs skip C-Score")
        print(f"     Miss rate:  {opt_mr:.1%} of skipped inputs had C-Score")
        print(f"                 flagged (false negatives)")
    else:
        print("  → No threshold found that satisfies the tolerance.")
        print("    C-Score may be catching things E+Q fundamentally cannot;")
        print("    consider always running C-Score.")

    # ── Agreement matrix stats ─────────────────────────────────────────
    if agreement is not None:
        print()
        print("-" * 65)
        print("  AGREEMENT MATRIX STATS")
        print("-" * 65)
        print(f"  True Positives:  {agreement['tp']:3d}   "
              f"False Positives: {agreement['fp']:3d}")
        print(f"  False Negatives: {agreement['fn']:3d}   "
              f"True Negatives:  {agreement['tn']:3d}")
        p = agreement["precision"]
        r = agreement["recall"]
        f1 = agreement["f1"]
        fnr = agreement["fnr"]
        print(f"  Precision: {p:.2f}  Recall: {r:.2f}  "
              f"F1: {f1:.2f}  FNR(miss): {fnr:.2%}")

    # ── Legacy threshold suggestion ────────────────────────────────────
    print()
    print("-" * 65)
    print("  SUGGESTED DECISION RULE")
    print("-" * 65)

    if n_additive == 0:
        print("  → C-Score never added value beyond E+Q in this corpus.")
        print("  → You may be able to skip C-Score entirely at all E+Q levels.")
    else:
        additive_eq_scores = [
            r["combined_eq_score"] for r in valid if r["c_score_additive"]
        ]
        max_additive_eq = max(additive_eq_scores)
        suggested_threshold = round(max_additive_eq + 0.05, 2)
        suggested_threshold = min(suggested_threshold, 1.0)

        print(f"  → C-Score added value for examples with max(E,Q) up to "
              f"{max_additive_eq:.2f}")
        print(f"  → Suggested skip threshold: max(E,Q) ≥ {suggested_threshold}")
        print(f"     (Skip C-Score when max(E,Q) ≥ {suggested_threshold})")
        print()
        print(f"  ⚠ With only {n} examples, this threshold is preliminary.")
        print(f"     Add more examples — especially edge cases — to refine it.")

    # C-Score factor breakdown
    print()
    print("-" * 65)
    print("  C-SCORE FACTOR BREAKDOWN (what the LLM uniquely detected)")
    print("-" * 65)

    factor_counts = {}
    factor_additive_counts = {}
    for r in valid:
        for fname, fdata in r.get("c_factors", {}).items():
            if fdata.get("detected"):
                factor_counts[fname] = factor_counts.get(fname, 0) + 1
                if r["c_score_additive"]:
                    factor_additive_counts[fname] = (
                        factor_additive_counts.get(fname, 0) + 1
                    )

    if factor_counts:
        print(f"  {'Factor':<35s}  {'Detected':>8s}  {'Additive':>8s}")
        print(f"  {'-'*35}  {'-'*8}  {'-'*8}")
        for fname in sorted(factor_counts, key=factor_counts.get, reverse=True):
            det = factor_counts[fname]
            add = factor_additive_counts.get(fname, 0)
            print(f"  {fname:<35s}  {det:>8d}  {add:>8d}")
    else:
        print("  (no factors detected across examples)")

    print()
    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="C-Score Necessity Calibration Benchmark")
    parser.add_argument("--limit", type=int, help="Limit the number of examples to run")
    args = parser.parse_args()

    # 1. Load examples
    examples = load_example_inputs()
    if args.limit:
        examples = examples[:args.limit]
    print(f"Loaded {len(examples)} example inputs.\n")

    # 2. Run scorers
    e_results = run_e_score(examples)
    q_results = run_q_score(examples)
    c_results = run_c_score(examples)

    # 3. Combine and derive
    combined = combine_results(examples, e_results, q_results, c_results)

    # 4. Save raw results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "benchmark_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "c_score_threshold": C_SCORE_THRESHOLD,
                    "num_examples": len(examples),
                },
                "results": combined,
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    print(f"Raw results saved to: {results_path}\n")

    # 5. Generate plots
    print("Generating plots...")
    plot_eq_scatter(combined, OUTPUT_DIR)
    plot_combined_vs_c(combined, OUTPUT_DIR)
    plot_heatmap(combined, OUTPUT_DIR)
    plot_per_example_bars(combined, OUTPUT_DIR)

    # 5b. Calibration analysis plots
    plot_threshold_sweep(combined, OUTPUT_DIR)
    corr_stats = plot_eq_c_correlation(combined, OUTPUT_DIR)

    # Find optimal threshold for the agreement matrix
    opt_t, _, _ = _find_optimal_threshold(combined)
    eq_thresh_for_matrix = opt_t if opt_t is not None else 0.5
    agreement = plot_agreement_matrix(combined, eq_thresh_for_matrix, OUTPUT_DIR)
    print()

    # 6. Summary (enhanced with calibration metrics)
    print_summary(combined, corr_stats=corr_stats, agreement=agreement)


if __name__ == "__main__":
    main()
