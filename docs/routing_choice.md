# Routing scheme decision

**Winner: gradient**

## Decision rule (pre-committed, `scenarios.yaml` §`decision_rule`)

> Pick the scheme with the highest PDR in the cell: `extra_obstruction_db=10`,
> `traffic_mix=mixed`, `mobility_m_per_s=1.0`, `n_nodes=7`,
> `offered_load_pkts_per_min_per_node=50`, `bs_role=mesh_participant`,
> `spacing_m=250`.
> Tiebreak 1: lower `control_overhead_ratio`. Tiebreak 2: lower P95 latency.

Source: `src/titan_communication/bakeoff/results/bakeoff_results.csv`
(58,320 runs total; 20 seeds × 3 schemes = 60 rows at the decision cell).

## Result at the decision cell

| Scheme   | N  | PDR (mean) | PDR (std) | Control overhead ratio | Latency P50 (ms) | Latency P95 (ms) |
|----------|----|-----------:|----------:|------------------------:|------------------:|------------------:|
| **gradient** | 20 | **0.2212** | 0.0247 | 0.1282 | 181.9 | 464.3 |
| flood    | 20 | 0.2060 | 0.0256 | 0.1732 | 174.2 | 410.8 |
| aodv     | 20 | 0.1158 | 0.0800 | 0.6067 | 165.3 | 402.5 |

**gradient** wins on the primary metric (PDR) alone — no tiebreak needed. For
what it's worth, it would also win tiebreak 1 (lowest control overhead: 0.1282
vs. flood's 0.1732); it does *not* win tiebreak 2 (P95 latency is its worst of
the three), but that's moot since it already wins on PDR.

aodv's very high `control_overhead_ratio` (0.61 — roughly one CTRL frame per
1.6 data frames) and high PDR variance (std 0.080, ~3× the other two schemes)
are consistent with RREQ/RREP churn under a mobile, moderately obstructed
7-node chain: routes break and get rediscovered often enough that overhead and
outcome both swing seed-to-seed.

## Robustness: channel-model sensitivity

The decision rule was evaluated once, under the base channel model in
`scenarios.yaml` (Petäjäjärvi 2015 suburban NLOS: PLE=2.7, ref. pathloss
40.6 dB, shadowing σ=7.8 dB). Because that constant choice is itself
uncertain relative to the Gazebo obstruction levels being modeled, we reran
the decision cell's scheme comparison — same traffic mix, mobility, node
count, load, BS role, spacing, 20 seeds — across obstruction levels
{0, 10, 20} dB under both the suburban model and a Petäjäjärvi urban NLOS
model (PLE=3.3, ref. pathloss 42.0 dB, shadowing σ=10.2 dB).

Script: `src/titan_communication/bakeoff/sensitivity.py`
Data: `eval/figures/bakeoff/sensitivity_summary.csv`
Plot: `eval/figures/bakeoff/sensitivity_pdr.pdf`

| Channel  | Obs (dB) | Winner | PDR | aodv / flood / gradient |
|----------|---------:|--------|----:|---------------------------|
| suburban | 0  | gradient | 0.2060 | 0.0245 / 0.1790 / 0.2060 |
| suburban | 10 | **gradient** | 0.2212 | 0.1158 / 0.2060 / 0.2212 |
| suburban | 20 | gradient | 0.2844 | 0.2656 / 0.2732 / 0.2844 |
| urban    | 0  | gradient | 0.2741 | 0.2618 / 0.2669 / 0.2741 |
| urban    | 10 | **aodv** | 0.3501 | 0.3501 / 0.3279 / 0.3300 |
| urban    | 20 | aodv | 0.2062 | 0.2062 / 0.1996 / 0.1617 |

**The decision is not fully robust to channel-model choice.** gradient wins
4 of 6 cells, including the pre-committed suburban/10 dB cell used for the
official decision. But under the urban model at 10 dB and 20 dB obstruction
— arguably the more relevant regime for a collapsed-structure SAR
deployment — aodv overtakes gradient by a similar margin. The suburban vs.
urban gap at 10 dB obstruction is large for every scheme (e.g. aodv:
0.1158 → 0.3501), so which Petäjäjärvi constants best match the actual
Gazebo obstruction model matters more to the outcome than the obstruction
level itself does.

**Open question, not yet resolved:** within a single channel model, PDR
*increases* with `extra_obstruction_db` (e.g. suburban aodv: 0.0245 at 0 dB
→ 0.2656 at 20 dB) rather than decreasing, which is not the expected
direction for added path loss. This trend is consistent across both channel
models and all three schemes, so it isn't a per-run fluke, but it hasn't
been root-caused (candidates: duty-cycle/backoff interaction at low
obstruction driving collisions that swamp the path-loss effect, or a
tick/timeout parameter that scales with something obstruction-correlated).
Until this is understood, the *magnitude* of PDR differences between
schemes/channels should be treated as more reliable than any claim about
the *direction* of the obstruction effect itself.

## Conclusion

Per the pre-committed decision rule, **gradient** is the selected routing
scheme. This holds under the base (suburban) channel model at every tested
obstruction level. It should be revisited if the urban Petäjäjärvi
constants turn out to be the better match for the target deployment
environment, since aodv overtakes gradient there at 10–20 dB obstruction —
and the unexplained obstruction-PDR trend above should be investigated
before either result is treated as final.
