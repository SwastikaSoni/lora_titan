"""
Sensitivity study: Petäjäjärvi 2015 urban vs suburban pathloss constants.

Reruns a subset of the sweep grid with two channel models and compares
PDR sensitivity to the choice. Answers the open question from README §8:
"Which Petäjäjärvi 2015 constants best match Gazebo obstruction levels?"

Channel models compared:
  - suburban: PLE=2.7, ref_PL=40.6, shadow=7.8 dB (default in scenarios.yaml)
  - urban:    PLE=3.3, ref_PL=42.0, shadow=10.2 dB (higher loss, more variability)

Subset: decision-rule cell params, swept across obstruction levels and
all three schemes. N=20 seeds. Much smaller than the full sweep.

Output:
  - eval/figures/bakeoff/sensitivity_pdr.pdf
  - eval/figures/bakeoff/sensitivity_summary.csv

Usage:
    cd ~/titan_ws
    python -m titan_communication.bakeoff.sensitivity \
        --config src/titan_communication/bakeoff/scenarios.yaml \
        --out eval/figures/bakeoff/
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.titan_communication.bakeoff.bakeoff import (
    load_config,
    make_lora_params,
    make_duty_policy,
    make_routing,
    run_single,
    SweepCell,
)


# ---------------------------------------------------------------------------
# Channel model variants
# ---------------------------------------------------------------------------

CHANNEL_VARIANTS = {
    "suburban": {
        "path_loss_exponent": 2.7,
        "reference_distance_m": 1.0,
        "reference_pathloss_db": 40.6,
        "shadowing_std_db": 7.8,
    },
    "urban": {
        "path_loss_exponent": 3.3,
        "reference_distance_m": 1.0,
        "reference_pathloss_db": 42.0,
        "shadowing_std_db": 10.2,
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Channel model sensitivity study")
    parser.add_argument(
        "--config", type=str,
        default="src/titan_communication/bakeoff/scenarios.yaml",
    )
    parser.add_argument(
        "--out", type=str,
        default="eval/figures/bakeoff/",
    )
    parser.add_argument(
        "--seeds", type=int, default=20,
        help="Seeds per cell",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    decision_cell = cfg["decision_rule"]["primary_cell"]
    schemes = cfg["schemes"]
    obstruction_levels = cfg["sweep"]["extra_obstruction_db"]
    seeds = list(range(1, args.seeds + 1))

    print(f"Sensitivity study: {len(CHANNEL_VARIANTS)} channel models × "
          f"{len(schemes)} schemes × {len(obstruction_levels)} obstruction × "
          f"{len(seeds)} seeds = "
          f"{len(CHANNEL_VARIANTS) * len(schemes) * len(obstruction_levels) * len(seeds)} runs")

    results = []
    total = len(CHANNEL_VARIANTS) * len(schemes) * len(obstruction_levels) * len(seeds)
    run_idx = 0
    t_start = time.monotonic()

    for ch_name, ch_params in CHANNEL_VARIANTS.items():
        for scheme_cfg in schemes:
            for obs in obstruction_levels:
                for seed in seeds:
                    run_idx += 1

                    # Build the cell matching the decision rule, but vary obstruction
                    cell = SweepCell(
                        n_nodes=decision_cell["n_nodes"],
                        traffic_mix_name=decision_cell["traffic_mix"],
                        traffic_classes=_get_traffic_classes(cfg, decision_cell["traffic_mix"]),
                        mobility_m_per_s=decision_cell["mobility_m_per_s"],
                        extra_obstruction_db=obs,
                        offered_load_pkts_per_min_per_node=decision_cell["offered_load_pkts_per_min_per_node"],
                        bs_role_name=decision_cell["bs_role"],
                        is_sink_only=_get_sink_flag(cfg, decision_cell["bs_role"]),
                        spacing_m=decision_cell["spacing_m"],
                    )

                    # Override the channel model in cfg for this run
                    cfg_copy = dict(cfg)
                    cfg_copy["channel"] = dict(ch_params)

                    elapsed = time.monotonic() - t_start
                    rate = elapsed / run_idx if run_idx > 0 else 0
                    eta = rate * (total - run_idx) / 60

                    print(
                        f"  [{run_idx}/{total}] {ch_name:10s} {scheme_cfg['name']:8s} "
                        f"obs={obs:2d} seed={seed:2d} ETA={eta:.1f}min",
                        end="", flush=True,
                    )

                    try:
                        result = run_single(scheme_cfg, cell, seed, cfg_copy)
                        result["channel_model"] = ch_name
                        results.append(result)
                        print(f"  PDR={result['pdr']:.4f}")
                    except Exception as e:
                        print(f"  FAILED: {e}")

    # ── Write CSV ──
    csv_path = out_dir / "sensitivity_summary.csv"
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved {csv_path}")

    # ── Plot ──
    _plot_sensitivity(results, out_dir)

    # ── Print summary ──
    _print_summary(results)


def _get_traffic_classes(cfg: dict, mix_name: str) -> dict[str, float]:
    for tmix in cfg["sweep"]["traffic_mix"]:
        if tmix["name"] == mix_name:
            return dict(tmix["classes"])
    return {"TELEM": 1.0}


def _get_sink_flag(cfg: dict, role_name: str) -> bool:
    for bsr in cfg["sweep"]["bs_role"]:
        if bsr["name"] == role_name:
            return bsr["is_sink_only"]
    return False


# ---------------------------------------------------------------------------
# Plot: PDR vs obstruction, one subplot per scheme, lines = channel model
# ---------------------------------------------------------------------------

def _plot_sensitivity(results: list[dict], out_dir: Path):
    if not results:
        return

    schemes = sorted(set(r["scheme"] for r in results))
    ch_models = sorted(set(r["channel_model"] for r in results))
    obs_vals = sorted(set(r["obstruction"] for r in results))

    ch_colors = {"suburban": "#2ca02c", "urban": "#d62728"}
    ch_markers = {"suburban": "o", "urban": "s"}

    fig, axes = plt.subplots(1, len(schemes), figsize=(3.5 * len(schemes), 3.0), sharey=True)
    if len(schemes) == 1:
        axes = [axes]

    for ax, scheme in zip(axes, schemes):
        for ch_name in ch_models:
            means = []
            ci_lo = []
            ci_hi = []
            for obs in obs_vals:
                vals = [r["pdr"] for r in results
                        if r["scheme"] == scheme
                        and r["channel_model"] == ch_name
                        and r["obstruction"] == obs]
                arr = np.array(vals) if vals else np.array([0.0])
                m = float(arr.mean())
                # Simple bootstrap
                if len(arr) > 1:
                    rng = np.random.default_rng(42)
                    boots = np.array([rng.choice(arr, len(arr), replace=True).mean()
                                      for _ in range(2000)])
                    lo = float(np.percentile(boots, 2.5))
                    hi = float(np.percentile(boots, 97.5))
                else:
                    lo, hi = m, m
                means.append(m)
                ci_lo.append(lo)
                ci_hi.append(hi)

            color = ch_colors.get(ch_name, "gray")
            marker = ch_markers.get(ch_name, "x")
            ax.errorbar(
                obs_vals, means,
                yerr=[np.array(means) - np.array(ci_lo),
                      np.array(ci_hi) - np.array(means)],
                label=ch_name, color=color, marker=marker,
                capsize=3, linewidth=1.2,
            )

        ax.set_xlabel("Extra obstruction (dB)")
        ax.set_title(scheme)
        ax.legend(fontsize=7)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("PDR")
    fig.suptitle("Sensitivity: suburban vs urban channel model", fontsize=10)
    fig.tight_layout()

    path = out_dir / "sensitivity_pdr.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict]):
    if not results:
        return

    print(f"\n{'='*70}")
    print("SENSITIVITY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Channel':10s} {'Scheme':10s} {'Obs':>4s} {'PDR_mean':>10s} {'PDR_std':>10s} "
          f"{'Lat50':>8s} {'N':>4s}")
    print("-" * 70)

    schemes = sorted(set(r["scheme"] for r in results))
    ch_models = sorted(set(r["channel_model"] for r in results))
    obs_vals = sorted(set(r["obstruction"] for r in results))

    for ch_name in ch_models:
        for scheme in schemes:
            for obs in obs_vals:
                vals = [r for r in results
                        if r["scheme"] == scheme
                        and r["channel_model"] == ch_name
                        and r["obstruction"] == obs]
                if not vals:
                    continue
                pdrs = np.array([r["pdr"] for r in vals])
                lats = np.array([r["latency_p50_ms"] for r in vals])
                print(f"{ch_name:10s} {scheme:10s} {obs:4.0f} "
                      f"{pdrs.mean():10.4f} {pdrs.std():10.4f} "
                      f"{lats.mean():8.1f} {len(vals):4d}")

    # Key finding for thesis
    print(f"\n{'='*70}")
    print("KEY FINDING:")

    for scheme in schemes:
        sub_pdrs = []
        urban_pdrs = []
        for r in results:
            if r["scheme"] == scheme and r["obstruction"] == 10:
                if r["channel_model"] == "suburban":
                    sub_pdrs.append(r["pdr"])
                elif r["channel_model"] == "urban":
                    urban_pdrs.append(r["pdr"])
        if sub_pdrs and urban_pdrs:
            diff = np.mean(sub_pdrs) - np.mean(urban_pdrs)
            print(f"  {scheme}: suburban PDR - urban PDR at 10dB obs = {diff:+.4f} "
                  f"({'suburban better' if diff > 0 else 'urban better'})")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()