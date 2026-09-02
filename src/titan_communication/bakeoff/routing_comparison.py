"""
Routing bake-off evaluation — reads bakeoff_results.csv and produces
the 2×2 comparison grid plus summary tables.

Plots (saved to eval/figures/bakeoff/):
  1. PDR vs extra_obstruction_db  (grouped by scheme, at decision-rule cell)
  2. Latency P50/P95 vs offered_load  (grouped by scheme)
  3. Control overhead ratio vs n_nodes  (grouped by scheme)
  4. Duty utilization vs mobility  (grouped by scheme)

Also prints:
  - Decision rule evaluation (which scheme wins)
  - Per-scheme summary table with bootstrap 95% CIs

Usage:
    cd ~/titan_ws
    python -m eval.bakeoff.routing_comparison \
        --csv src/titan_communication/bakeoff/results/bakeoff_results.csv \
        --out eval/figures/bakeoff/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# IEEE column width sizing (matches eval/common/plot_style.py intent)
# ---------------------------------------------------------------------------
FIGSIZE_2x2 = (7.16, 6.0)    # IEEE double-column width, ~square
FIGSIZE_1x1 = (3.5, 2.8)     # IEEE single-column
DPI = 300
SCHEME_COLORS = {"flood": "#1f77b4", "aodv": "#ff7f0e", "gradient": "#2ca02c"}
SCHEME_MARKERS = {"flood": "o", "aodv": "s", "gradient": "^"}


def set_style():
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(data: np.ndarray, n_boot: int = 5000, ci: float = 0.95) -> tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high) via bootstrap resampling."""
    if len(data) == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(42)
    means = np.array([
        rng.choice(data, size=len(data), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(means, alpha * 100))
    hi = float(np.percentile(means, (1 - alpha) * 100))
    return (float(data.mean()), lo, hi)


# ---------------------------------------------------------------------------
# Plot 1: PDR vs obstruction
# ---------------------------------------------------------------------------

def plot_pdr_vs_obstruction(df: pd.DataFrame, out_dir: Path,
                            fixed: dict):
    """PDR vs extra_obstruction_db, one line per scheme.
    Fixed axes: n_nodes, traffic_mix, mobility, load, bs_role from decision rule."""
    mask = (
        (df["n_nodes"] == fixed["n_nodes"]) &
        (df["traffic_mix"] == fixed["traffic_mix"]) &
        (df["mobility"] == fixed["mobility"]) &
        (df["load"] == fixed["load"]) &
        (df["bs_role"] == fixed["bs_role"]) &
        (df["spacing_m"] == fixed["spacing_m"])
    )
    sub = df[mask]

    fig, ax = plt.subplots(figsize=FIGSIZE_1x1)

    obs_vals = sorted(sub["obstruction"].unique())

    for scheme in sub["scheme"].unique():
        means, lows, highs = [], [], []
        for obs in obs_vals:
            vals = sub[(sub["scheme"] == scheme) & (sub["obstruction"] == obs)]["pdr"].values
            m, lo, hi = bootstrap_ci(vals)
            means.append(m)
            lows.append(lo)
            highs.append(hi)

        color = SCHEME_COLORS.get(scheme, "gray")
        marker = SCHEME_MARKERS.get(scheme, "x")
        ax.errorbar(
            obs_vals, means,
            yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
            label=scheme, color=color, marker=marker, capsize=3, linewidth=1.2,
        )

    ax.set_xlabel("Extra obstruction (dB)")
    ax.set_ylabel("Packet Delivery Ratio")
    ax.set_title(f"PDR vs obstruction\n(n={fixed['n_nodes']}, {fixed['traffic_mix']}, "
                 f"mob={fixed['mobility']}m/s, load={fixed['load']})")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    path = out_dir / "pdr_vs_obstruction.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 2: Latency vs offered load
# ---------------------------------------------------------------------------

def plot_latency_vs_load(df: pd.DataFrame, out_dir: Path, fixed: dict):
    """Latency P50 and P95 vs offered_load, one line per scheme."""
    mask = (
        (df["n_nodes"] == fixed["n_nodes"]) &
        (df["traffic_mix"] == fixed["traffic_mix"]) &
        (df["mobility"] == fixed["mobility"]) &
        (df["obstruction"] == fixed["obstruction"]) &
        (df["bs_role"] == fixed["bs_role"]) &
        (df["spacing_m"] == fixed["spacing_m"])
    )
    sub = df[mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)

    load_vals = sorted(sub["load"].unique())

    for scheme in sub["scheme"].unique():
        color = SCHEME_COLORS.get(scheme, "gray")
        marker = SCHEME_MARKERS.get(scheme, "x")

        for ax, col, label_suffix in [
            (ax1, "latency_p50_ms", "P50"),
            (ax2, "latency_p95_ms", "P95"),
        ]:
            means, lows, highs = [], [], []
            for ld in load_vals:
                vals = sub[(sub["scheme"] == scheme) & (sub["load"] == ld)][col].values
                m, lo, hi = bootstrap_ci(vals)
                means.append(m)
                lows.append(lo)
                highs.append(hi)

            ax.errorbar(
                load_vals, means,
                yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
                label=f"{scheme}", color=color, marker=marker, capsize=3, linewidth=1.2,
            )

    ax1.set_xlabel("Offered load (pkts/min/node)")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Latency P50")
    ax1.legend()

    ax2.set_xlabel("Offered load (pkts/min/node)")
    ax2.set_title("Latency P95")
    ax2.legend()

    fig.suptitle(f"Latency vs load (n={fixed['n_nodes']}, {fixed['traffic_mix']}, "
                 f"obs={fixed['obstruction']}dB)", fontsize=9)
    fig.tight_layout()

    path = out_dir / "latency_vs_load.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 3: Control overhead vs n_nodes
# ---------------------------------------------------------------------------

def plot_overhead_vs_nodes(df: pd.DataFrame, out_dir: Path, fixed: dict):
    """Control overhead ratio vs n_nodes, one line per scheme."""
    mask = (
        (df["traffic_mix"] == fixed["traffic_mix"]) &
        (df["mobility"] == fixed["mobility"]) &
        (df["obstruction"] == fixed["obstruction"]) &
        (df["load"] == fixed["load"]) &
        (df["bs_role"] == fixed["bs_role"]) &
        (df["spacing_m"] == fixed["spacing_m"])
    )
    sub = df[mask]

    fig, ax = plt.subplots(figsize=FIGSIZE_1x1)
    node_vals = sorted(sub["n_nodes"].unique())

    for scheme in sub["scheme"].unique():
        color = SCHEME_COLORS.get(scheme, "gray")
        marker = SCHEME_MARKERS.get(scheme, "x")
        means, lows, highs = [], [], []
        for n in node_vals:
            vals = sub[(sub["scheme"] == scheme) & (sub["n_nodes"] == n)]["control_overhead_ratio"].values
            m, lo, hi = bootstrap_ci(vals)
            means.append(m)
            lows.append(lo)
            highs.append(hi)

        ax.errorbar(
            node_vals, means,
            yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
            label=scheme, color=color, marker=marker, capsize=3, linewidth=1.2,
        )

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Control overhead ratio")
    ax.set_title(f"Control overhead vs network size\n({fixed['traffic_mix']}, "
                 f"obs={fixed['obstruction']}dB, load={fixed['load']})")
    ax.legend()

    path = out_dir / "overhead_vs_nodes.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 4: Duty utilization vs mobility
# ---------------------------------------------------------------------------

def plot_duty_vs_mobility(df: pd.DataFrame, out_dir: Path, fixed: dict):
    """Mean duty utilization vs mobility, one line per scheme."""
    mask = (
        (df["n_nodes"] == fixed["n_nodes"]) &
        (df["traffic_mix"] == fixed["traffic_mix"]) &
        (df["obstruction"] == fixed["obstruction"]) &
        (df["load"] == fixed["load"]) &
        (df["bs_role"] == fixed["bs_role"]) &
        (df["spacing_m"] == fixed["spacing_m"])
    )
    sub = df[mask]

    fig, ax = plt.subplots(figsize=FIGSIZE_1x1)
    mob_vals = sorted(sub["mobility"].unique())

    for scheme in sub["scheme"].unique():
        color = SCHEME_COLORS.get(scheme, "gray")
        marker = SCHEME_MARKERS.get(scheme, "x")
        means, lows, highs = [], [], []
        for mob in mob_vals:
            vals = sub[(sub["scheme"] == scheme) & (sub["mobility"] == mob)]["duty_utilization_mean"].values
            m, lo, hi = bootstrap_ci(vals)
            means.append(m)
            lows.append(lo)
            highs.append(hi)

        ax.errorbar(
            mob_vals, means,
            yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
            label=scheme, color=color, marker=marker, capsize=3, linewidth=1.2,
        )

    ax.set_xlabel("Mobility (m/s)")
    ax.set_ylabel("Mean duty utilization")
    ax.set_title(f"Duty utilization vs mobility\n(n={fixed['n_nodes']}, "
                 f"{fixed['traffic_mix']}, obs={fixed['obstruction']}dB)")
    ax.legend()

    path = out_dir / "duty_vs_mobility.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Combined 2x2 grid
# ---------------------------------------------------------------------------

def plot_combined_grid(df: pd.DataFrame, out_dir: Path, fixed: dict):
    """All four plots in a single 2x2 figure."""
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_2x2)
    schemes = sorted(df["scheme"].unique())

    # ── (0,0) PDR vs obstruction ──
    ax = axes[0, 0]
    mask = (
        (df["n_nodes"] == fixed["n_nodes"]) &
        (df["traffic_mix"] == fixed["traffic_mix"]) &
        (df["mobility"] == fixed["mobility"]) &
        (df["load"] == fixed["load"]) &
        (df["bs_role"] == fixed["bs_role"]) &
        (df["spacing_m"] == fixed["spacing_m"])
    )
    sub = df[mask]
    obs_vals = sorted(sub["obstruction"].unique())
    for scheme in schemes:
        color = SCHEME_COLORS.get(scheme, "gray")
        marker = SCHEME_MARKERS.get(scheme, "x")
        means = []
        for obs in obs_vals:
            vals = sub[(sub["scheme"] == scheme) & (sub["obstruction"] == obs)]["pdr"].values
            means.append(vals.mean() if len(vals) > 0 else 0)
        ax.plot(obs_vals, means, label=scheme, color=color, marker=marker, linewidth=1.2)
    ax.set_xlabel("Extra obstruction (dB)")
    ax.set_ylabel("PDR")
    ax.set_title("(a) PDR vs obstruction")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6)

    # ── (0,1) Latency P50/P95 vs load ──
    ax = axes[0, 1]
    mask2 = (
        (df["n_nodes"] == fixed["n_nodes"]) &
        (df["traffic_mix"] == fixed["traffic_mix"]) &
        (df["mobility"] == fixed["mobility"]) &
        (df["obstruction"] == fixed["obstruction"]) &
        (df["bs_role"] == fixed["bs_role"]) &
        (df["spacing_m"] == fixed["spacing_m"])
    )
    sub2 = df[mask2]
    load_vals = sorted(sub2["load"].unique())
    for scheme in schemes:
        color = SCHEME_COLORS.get(scheme, "gray")
        p50s, p95s = [], []
        for ld in load_vals:
            vals = sub2[(sub2["scheme"] == scheme) & (sub2["load"] == ld)]
            p50s.append(vals["latency_p50_ms"].mean() if len(vals) > 0 else 0)
            p95s.append(vals["latency_p95_ms"].mean() if len(vals) > 0 else 0)
        ax.plot(load_vals, p50s, label=f"{scheme} P50", color=color, marker="o",
                linewidth=1.2, linestyle="-")
        ax.plot(load_vals, p95s, label=f"{scheme} P95", color=color, marker="^",
                linewidth=1.0, linestyle="--")
    ax.set_xlabel("Load (pkts/min/node)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("(b) Latency vs load")
    ax.legend(fontsize=5, ncol=2)

    # ── (1,0) Control overhead vs n_nodes ──
    ax = axes[1, 0]
    mask3 = (
        (df["traffic_mix"] == fixed["traffic_mix"]) &
        (df["mobility"] == fixed["mobility"]) &
        (df["obstruction"] == fixed["obstruction"]) &
        (df["load"] == fixed["load"]) &
        (df["bs_role"] == fixed["bs_role"]) &
        (df["spacing_m"] == fixed["spacing_m"])
    )
    sub3 = df[mask3]
    node_vals = sorted(sub3["n_nodes"].unique())
    for scheme in schemes:
        color = SCHEME_COLORS.get(scheme, "gray")
        marker = SCHEME_MARKERS.get(scheme, "x")
        means = []
        for n in node_vals:
            vals = sub3[(sub3["scheme"] == scheme) & (sub3["n_nodes"] == n)]["control_overhead_ratio"].values
            means.append(vals.mean() if len(vals) > 0 else 0)
        ax.plot(node_vals, means, label=scheme, color=color, marker=marker, linewidth=1.2)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Control overhead ratio")
    ax.set_title("(c) Overhead vs network size")
    ax.legend(fontsize=6)

    # ── (1,1) Duty utilization vs mobility ──
    ax = axes[1, 1]
    mask4 = (
        (df["n_nodes"] == fixed["n_nodes"]) &
        (df["traffic_mix"] == fixed["traffic_mix"]) &
        (df["obstruction"] == fixed["obstruction"]) &
        (df["load"] == fixed["load"]) &
        (df["bs_role"] == fixed["bs_role"]) &
        (df["spacing_m"] == fixed["spacing_m"])
    )
    sub4 = df[mask4]
    mob_vals = sorted(sub4["mobility"].unique())
    for scheme in schemes:
        color = SCHEME_COLORS.get(scheme, "gray")
        marker = SCHEME_MARKERS.get(scheme, "x")
        means = []
        for mob in mob_vals:
            vals = sub4[(sub4["scheme"] == scheme) & (sub4["mobility"] == mob)]["duty_utilization_mean"].values
            means.append(vals.mean() if len(vals) > 0 else 0)
        ax.plot(mob_vals, means, label=scheme, color=color, marker=marker, linewidth=1.2)
    ax.set_xlabel("Mobility (m/s)")
    ax.set_ylabel("Duty utilization")
    ax.set_title("(d) Duty vs mobility")
    ax.legend(fontsize=6)

    fig.tight_layout()
    path = out_dir / "routing_comparison_2x2.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Decision rule evaluation
# ---------------------------------------------------------------------------

def evaluate_decision_rule(df: pd.DataFrame, decision_cfg: dict):
    """Apply the pre-committed decision rule and print the winner."""
    cell = decision_cfg["primary_cell"]

    mask = (
        (df["n_nodes"] == cell["n_nodes"]) &
        (df["traffic_mix"] == cell["traffic_mix"]) &
        (df["mobility"] == cell["mobility_m_per_s"]) &
        (df["obstruction"] == cell["extra_obstruction_db"]) &
        (df["load"] == cell["offered_load_pkts_per_min_per_node"]) &
        (df["bs_role"] == cell["bs_role"]) &
        (df["spacing_m"] == cell["spacing_m"])
    )
    sub = df[mask]

    if sub.empty:
        print("\n  WARNING: Decision rule cell not found in results.")
        print(f"  Looking for: {cell}")
        print(f"  Available n_nodes: {sorted(df['n_nodes'].unique())}")
        print(f"  Available schemes: {sorted(df['scheme'].unique())}")
        return None

    print(f"\n{'='*65}")
    print("DECISION RULE EVALUATION")
    print(f"{'='*65}")
    print(f"Cell: n={cell['n_nodes']}, mix={cell['traffic_mix']}, "
          f"mob={cell['mobility_m_per_s']}, obs={cell['extra_obstruction_db']}, "
          f"load={cell['offered_load_pkts_per_min_per_node']}, bs={cell['bs_role']}")
    print()

    results = []
    for scheme in sorted(sub["scheme"].unique()):
        s = sub[sub["scheme"] == scheme]
        pdr_vals = s["pdr"].values
        overhead_vals = s["control_overhead_ratio"].values
        lat95_vals = s["latency_p95_ms"].values

        pdr_m, pdr_lo, pdr_hi = bootstrap_ci(pdr_vals)
        oh_m, oh_lo, oh_hi = bootstrap_ci(overhead_vals)
        lat_m, lat_lo, lat_hi = bootstrap_ci(lat95_vals)

        results.append({
            "scheme": scheme,
            "pdr_mean": pdr_m, "pdr_ci": (pdr_lo, pdr_hi),
            "overhead_mean": oh_m, "overhead_ci": (oh_lo, oh_hi),
            "lat95_mean": lat_m, "lat95_ci": (lat_lo, lat_hi),
            "n_seeds": len(s),
        })

        print(f"  {scheme:10s}  PDR={pdr_m:.4f} [{pdr_lo:.4f}, {pdr_hi:.4f}]  "
              f"OH={oh_m:.4f}  Lat95={lat_m:.1f}ms  (N={len(s)})")

    # Sort by decision rule: primary=PDR (desc), tiebreak1=overhead (asc), tiebreak2=lat95 (asc)
    results.sort(key=lambda r: (-r["pdr_mean"], r["overhead_mean"], r["lat95_mean"]))
    winner = results[0]["scheme"]

    print()
    print(f"  WINNER: {winner}")
    print(f"    Primary metric (PDR):        {results[0]['pdr_mean']:.4f}")
    print(f"    Tiebreak 1 (overhead):       {results[0]['overhead_mean']:.4f}")
    print(f"    Tiebreak 2 (latency P95):    {results[0]['lat95_mean']:.1f} ms")
    print(f"{'='*65}")

    return winner


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(df: pd.DataFrame):
    """Print a per-scheme aggregate summary."""
    print(f"\n{'='*65}")
    print("PER-SCHEME AGGREGATE SUMMARY (all cells)")
    print(f"{'='*65}")
    print(f"{'Scheme':10s} {'PDR':>8s} {'Lat50':>8s} {'Lat95':>8s} "
          f"{'OH':>8s} {'Duty':>8s} {'N':>6s}")
    print("-" * 65)

    for scheme in sorted(df["scheme"].unique()):
        s = df[df["scheme"] == scheme]
        print(f"{scheme:10s} "
              f"{s['pdr'].mean():8.4f} "
              f"{s['latency_p50_ms'].mean():8.1f} "
              f"{s['latency_p95_ms'].mean():8.1f} "
              f"{s['control_overhead_ratio'].mean():8.4f} "
              f"{s['duty_utilization_mean'].mean():8.4f} "
              f"{len(s):6d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Routing bake-off evaluation")
    parser.add_argument(
        "--csv", type=str,
        default="src/titan_communication/bakeoff/results/bakeoff_results.csv",
    )
    parser.add_argument(
        "--out", type=str,
        default="eval/figures/bakeoff/",
    )
    parser.add_argument(
        "--decision-yaml", type=str,
        default="src/titan_communication/bakeoff/scenarios.yaml",
        help="scenarios.yaml containing the decision_rule block",
    )
    args = parser.parse_args()

    set_style()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv)
    print(f"  {len(df)} rows, schemes: {sorted(df['scheme'].unique())}")

    # Load decision rule from scenarios.yaml
    import yaml
    scfg = yaml.safe_load(Path(args.decision_yaml).read_text())
    decision_cfg = scfg.get("decision_rule", {})
    decision_cell = decision_cfg.get("primary_cell", {})

    # Fixed params for plots (from decision rule cell, with defaults)
    fixed = {
        "n_nodes": decision_cell.get("n_nodes", 7),
        "traffic_mix": decision_cell.get("traffic_mix", "mixed"),
        "mobility": decision_cell.get("mobility_m_per_s", 1.0),
        "obstruction": decision_cell.get("extra_obstruction_db", 10),
        "load": decision_cell.get("offered_load_pkts_per_min_per_node", 50),
        "bs_role": decision_cell.get("bs_role", "mesh_participant"),
        "spacing_m": decision_cell.get("spacing_m", 250),
    }

    print(f"\nGenerating plots (fixed axes from decision rule cell)...")
    print(f"  Fixed: {fixed}")

    plot_pdr_vs_obstruction(df, out_dir, fixed)
    plot_latency_vs_load(df, out_dir, fixed)
    plot_overhead_vs_nodes(df, out_dir, fixed)
    plot_duty_vs_mobility(df, out_dir, fixed)
    plot_combined_grid(df, out_dir, fixed)

    print_summary_table(df)
    winner = evaluate_decision_rule(df, decision_cfg)

    if winner:
        # Write decision to markdown
        doc_path = Path("docs/routing_choice.md")
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        with open(doc_path, "w") as f:
            f.write(f"# Routing scheme decision\n\n")
            f.write(f"**Winner: {winner}**\n\n")
            f.write(f"Decision rule cell:\n")
            for k, v in decision_cell.items():
                f.write(f"  - {k}: {v}\n")
            f.write(f"\nSelected by: highest PDR at the decision cell, "
                    f"tiebreak on control overhead, then P95 latency.\n")
            f.write(f"\nResults from {len(df)} runs across {len(df['scheme'].unique())} schemes.\n")
        print(f"\n  Decision documented in {doc_path}")


if __name__ == "__main__":
    main()