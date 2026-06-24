"""
C-Score Necessity Calibration Benchmark
========================================

Runs all three scorers (E-Score, Q-Score, C-Score) on example inputs,
compares their outputs, and generates plots + analysis to determine
when the expensive C-Score (LLM) can safely be skipped.

Usage:
    python benchmark.py
"""

import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from example_inputs import load_example_inputs

# ---------------------------------------------------------------------------
# Scorer imports
# ---------------------------------------------------------------------------
from importlib import import_module


def _import_e_scorer():
    """Import and return the E-Score detector class."""
    mod = import_module("e-score")
    return mod.ExplicitPIIDetector


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
OUTPUT_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Running the scorers
# ---------------------------------------------------------------------------

def run_e_score(examples: list[dict]) -> list[dict]:
    """Run the E-Score detector on all examples."""
    print("=" * 60)
    print("Running E-Score (Presidio PII/PHI detection)...")
    print("=" * 60)

    ExplicitPIIDetector = _import_e_scorer()
    detector = ExplicitPIIDetector()

    results = []
    for i, ex in enumerate(examples, 1):
        name = ex.get("name", f"Example {i}")
        result = detector.analyze(ex["text"])
        results.append({
            "name": name,
            "e_score": result.e_score,
            "entity_count": result.entity_count,
            "has_critical_pii": result.has_critical_pii,
            "max_weighted_score": result.max_weighted_score,
            "total_weighted_score": result.total_weighted_score,
            "entities": [
                {
                    "type": e.entity_type,
                    "text": e.text,
                    "confidence": e.confidence,
                    "sensitivity_weight": e.sensitivity_weight,
                    "weighted_score": e.weighted_score,
                }
                for e in result.entities
            ],
        })
        print(f"  [{i}/{len(examples)}] {name}: e_score={result.e_score:.4f}  "
              f"entities={result.entity_count}")

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
            "q_rarest": q_result.q_rarest,
            "q_combination": q_result.q_combination,
            "q_subsets": q_result.q_subsets,
            "expected_k": q_result.expected_k,
            "num_qis": len(q_result.detected_qis),
            "identifying_subsets_count": len(q_result.identifying_subsets),
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
            "c_num_factors": c["num_factors_detected"],

            # Extra detail
            "e_has_critical_pii": e["has_critical_pii"],
            "q_expected_k": round(q["expected_k"], 4),
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


# ---------------------------------------------------------------------------
# Summary analysis
# ---------------------------------------------------------------------------

def print_summary(combined: list[dict]):
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

    # Threshold suggestion
    print()
    print("-" * 65)
    print("  SUGGESTED DECISION RULE")
    print("-" * 65)

    # Find the lowest combined_eq_score where C-Score was NOT additive
    if n_additive == 0:
        print("  → C-Score never added value beyond E+Q in this corpus.")
        print("  → You may be able to skip C-Score entirely at all E+Q levels.")
    else:
        # Find the threshold
        additive_eq_scores = [
            r["combined_eq_score"] for r in valid if r["c_score_additive"]
        ]
        max_additive_eq = max(additive_eq_scores)
        suggested_threshold = round(max_additive_eq + 0.05, 2)  # small margin
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
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  C-Score Necessity Calibration Benchmark                   ║")
    print("║  Determines when E+Q scores are sufficient to skip the LLM ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # 1. Load examples
    examples = load_example_inputs()
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
    print()

    # 6. Summary
    print_summary(combined)


if __name__ == "__main__":
    main()
