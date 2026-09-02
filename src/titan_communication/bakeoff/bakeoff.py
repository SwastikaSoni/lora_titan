"""
Titan DMS routing bake-off harness.

Reads scenarios.yaml, runs the full cartesian product sweep, writes
one CSV row per completed run. Resume-safe: restarts skip rows already
in the CSV.

Usage:
    cd ~/titan_ws
    python -m titan_communication.bakeoff.bakeoff \
        --config src/titan_communication/bakeoff/scenarios.yaml \
        --out    src/titan_communication/bakeoff/results/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import itertools
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import simpy
import yaml

# ── Project imports ──
from titan_communication.radio.airtime import (
    CodingRate,
    LoRaParams,
    time_on_air_s,
)
from titan_communication.radio.channel import (
    ChannelModel,
    PathLossModel,
    ShadowingModel,
)
from titan_communication.radio.duty import DutyBucket, RegionPolicy
from titan_communication.radio.virtual import VirtualChannel, VirtualLoRaTransport
from titan_communication.mesh.frame import Frame, MessageClass, BROADCAST_ID
from titan_communication.mesh.queue import PriorityQueue
from titan_communication.mesh.stack import MeshStack

# topology.py lives in the same bakeoff/ directory
from .topology import Topology


# =====================================================================
# Config loading
# =====================================================================

def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def make_lora_params(cfg: dict) -> LoRaParams:
    lora = cfg["lora"]
    return LoRaParams(
        spreading_factor=int(lora["spreading_factor"]),
        bandwidth_hz=int(lora["bandwidth_hz"]),
        coding_rate=CodingRate.from_str(str(lora["coding_rate"])),
        preamble_symbols=int(lora.get("preamble_symbols", 8)),
        explicit_header=bool(lora.get("explicit_header", True)),
        crc_on=bool(lora.get("crc_on", True)),
    )


def make_duty_policy(cfg: dict) -> RegionPolicy:
    name = cfg["duty"]["policy"]
    factories = {
        "us915_polite": RegionPolicy.us915_polite,
        "us915_fcc_raw": RegionPolicy.us915_fcc_raw,
        "us915_hopping": RegionPolicy.us915_hopping,
        "in865": RegionPolicy.in865,
        "eu868": RegionPolicy.eu868,
    }
    return factories[name]()


def make_channel_model(cfg: dict, extra_obstruction_db: float) -> ChannelModel:
    ch = cfg["channel"]
    return ChannelModel(
        name=f"bakeoff_obs{extra_obstruction_db}",
        pathloss=PathLossModel(
            path_loss_exponent=float(ch["path_loss_exponent"]),
            reference_distance_m=float(ch["reference_distance_m"]),
            reference_pathloss_db=float(ch["reference_pathloss_db"]) + extra_obstruction_db,
        ),
        shadowing=ShadowingModel(std_dev_db=float(ch["shadowing_std_db"])),
    )


def make_routing(scheme_cfg: dict, node_addr: int, bs_addr: int, is_bs: bool):
    """Dynamically import and instantiate the routing class."""
    mod = importlib.import_module(scheme_cfg["module"])
    cls = getattr(mod, scheme_cfg["class"])
    return cls(
        node_addr=node_addr,
        bs_addr=bs_addr,
        is_bs=is_bs,
        config=dict(scheme_cfg.get("config", {})),
    )


# =====================================================================
# Sweep generation
# =====================================================================

@dataclass(frozen=True)
class SweepCell:
    """One combination of sweep parameters."""
    n_nodes: int
    traffic_mix_name: str
    traffic_classes: dict[str, float]
    mobility_m_per_s: float
    extra_obstruction_db: float
    offered_load_pkts_per_min_per_node: float
    bs_role_name: str
    is_sink_only: bool
    spacing_m: float

    @property
    def key_tuple(self) -> tuple:
        return (
            self.n_nodes, self.traffic_mix_name, self.mobility_m_per_s,
            self.extra_obstruction_db, self.offered_load_pkts_per_min_per_node,
            self.bs_role_name, self.spacing_m,
        )


def generate_sweep(cfg: dict) -> list[SweepCell]:
    sw = cfg["sweep"]
    cells = []
    for (n, tmix, mob, obs, load, bsr, spacing) in itertools.product(
        sw["n_nodes"],
        sw["traffic_mix"],
        sw["mobility_m_per_s"],
        sw["extra_obstruction_db"],
        sw["offered_load_pkts_per_min_per_node"],
        sw["bs_role"],
        sw["spacing_m"],
    ):
        cells.append(SweepCell(
            n_nodes=n,
            traffic_mix_name=tmix["name"],
            traffic_classes=dict(tmix["classes"]),
            mobility_m_per_s=mob,
            extra_obstruction_db=obs,
            offered_load_pkts_per_min_per_node=load,
            bs_role_name=bsr["name"],
            is_sink_only=bsr["is_sink_only"],
            spacing_m=spacing,
        ))
    return cells


# =====================================================================
# Resume support
# =====================================================================

CSV_COLUMNS = [
    "scheme", "seed", "n_nodes", "traffic_mix", "mobility",
    "obstruction", "load", "bs_role", "spacing_m",
    # metrics
    "pdr", "latency_p50_ms", "latency_p95_ms",
    "control_overhead_ratio", "duty_utilization_mean",
    "tx_total", "rx_total", "deliver_total", "drop_total",
    "forward_total", "busy_wait_total",
    "wall_clock_s",
]


def _run_key(scheme_name: str, seed: int, cell: SweepCell) -> tuple:
    return (
        scheme_name, seed, cell.n_nodes, cell.traffic_mix_name,
        cell.mobility_m_per_s, cell.extra_obstruction_db,
        cell.offered_load_pkts_per_min_per_node, cell.bs_role_name,
        cell.spacing_m,
    )


def load_completed(csv_path: Path) -> set[tuple]:
    done: set[tuple] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (
                    row["scheme"],
                    int(row["seed"]),
                    int(row["n_nodes"]),
                    row["traffic_mix"],
                    float(row["mobility"]),
                    float(row["obstruction"]),
                    float(row["load"]),
                    row["bs_role"],
                    float(row["spacing_m"]),
                )
                done.add(key)
            except (KeyError, ValueError):
                continue
    return done


# =====================================================================
# Mobile position helper
# =====================================================================

class MobilePosition:
    """Mutable position callable for VirtualLoRaTransport position_getter."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __call__(self) -> tuple[float, float]:
        return (self.x, self.y)


def mobility_process(
    env: simpy.Environment,
    positions: list[MobilePosition],
    speed_m_per_s: float,
    area_half_width: float,
    rng: np.random.Generator,
    dt_s: float = 1.0,
):
    """SimPy process: each mobile position moves toward random waypoints."""
    if speed_m_per_s <= 0:
        return

    # Initial waypoints for each mobile
    n = len(positions)
    waypoints = [
        (float(rng.uniform(-area_half_width, area_half_width)),
         float(rng.uniform(-area_half_width, area_half_width)))
        for _ in range(n)
    ]

    while True:
        yield env.timeout(dt_s)
        step = speed_m_per_s * dt_s

        for i, pos in enumerate(positions):
            wx, wy = waypoints[i]
            dx = wx - pos.x
            dy = wy - pos.y
            dist = math.hypot(dx, dy)

            if dist <= step:
                # Reached waypoint — pick a new one
                pos.x = wx
                pos.y = wy
                waypoints[i] = (
                    float(rng.uniform(-area_half_width, area_half_width)),
                    float(rng.uniform(-area_half_width, area_half_width)),
                )
            else:
                # Move toward waypoint
                ratio = step / dist
                pos.x += dx * ratio
                pos.y += dy * ratio


# =====================================================================
# PDR + latency tracker
# =====================================================================

@dataclass
class SentRecord:
    src_id: int
    seq: int
    dst_id: int
    send_time: float
    msg_class: MessageClass


# ---------------------------------------------------------------------------
# Tracker — fixed: uses payload-embedded tracking key
# ---------------------------------------------------------------------------

class Tracker:
    """Tracks sent frames and delivered frames for PDR/latency computation."""

    def __init__(self, warmup_s: float = 0.0):
        self.sent: list[SentRecord] = []
        # key = "robot_id:app_seq" string embedded in payload
        self.delivered: dict[str, float] = {}
        self.warmup_s = warmup_s

    def record_send(self, src_id: int, app_seq: int, dst_id: int,
                    send_time: float, msg_class: MessageClass) -> None:
        self.sent.append(SentRecord(src_id, app_seq, dst_id, send_time, msg_class))

    def record_rx_at_bs(self, frame: Frame, rx_time: float) -> None:
        """Called from BS sniffer. Extracts tracking key from payload."""
        # Skip control frames — they're not app data
        if frame.message_class == MessageClass.CTRL:
            return

        try:
            # Payload is b"robot_id:app_seq" as set by _robot_send_loop
            tracking_key = frame.payload.decode("utf-8", errors="ignore")
        except Exception:
            return

        if tracking_key and tracking_key not in self.delivered:
            self.delivered[tracking_key] = rx_time

    def compute_metrics(self) -> dict:
        valid_sent = [s for s in self.sent if s.send_time >= self.warmup_s]

        if not valid_sent:
            return {"pdr": 0.0, "latency_p50_ms": 0.0, "latency_p95_ms": 0.0}

        latencies: list[float] = []
        delivered_count = 0
        for s in valid_sent:
            tracking_key = f"{s.src_id}:{s.seq}"
            if tracking_key in self.delivered:
                delivered_count += 1
                lat_ms = (self.delivered[tracking_key] - s.send_time) * 1000.0
                latencies.append(lat_ms)

        pdr = delivered_count / len(valid_sent) if valid_sent else 0.0

        if latencies:
            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
            p95 = latencies[p95_idx]
        else:
            p50 = 0.0
            p95 = 0.0

        return {
            "pdr": round(pdr, 6),
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
        }


# =====================================================================
# Traffic generator (SimPy process)
# =====================================================================

def traffic_generator(
    env: simpy.Environment,
    stacks: dict[int, MeshStack],
    tracker: Tracker,
    bs_id: int,
    offered_load_pkts_per_min_per_node: float,
    traffic_classes: dict[str, float],
    rng: np.random.Generator,
):
    """Spawn one independent send-loop per robot."""
    if offered_load_pkts_per_min_per_node <= 0:
        return

    robot_ids = [nid for nid in stacks if nid != bs_id]

    # Build CDF for traffic class selection
    class_names = list(traffic_classes.keys())
    class_weights = [traffic_classes[c] for c in class_names]
    total_w = sum(class_weights)
    class_weights = [w / total_w for w in class_weights]

    msg_class_map = {
        "SOS": MessageClass.SOS,
        "CTRL": MessageClass.CTRL,
        "TELEM": MessageClass.TELEM,
        "AUDIO": MessageClass.AUDIO,
        "BULK": MessageClass.BULK,
    }

    interval_s = 60.0 / offered_load_pkts_per_min_per_node

    for rid in robot_ids:
        # Each robot gets its own independent RNG stream
        child_rng = np.random.default_rng(rng.integers(0, 2**31))
        env.process(_robot_send_loop(
            env=env,
            stack=stacks[rid],
            tracker=tracker,
            robot_id=rid,
            bs_id=bs_id,
            interval_s=interval_s,
            class_names=class_names,
            class_weights=class_weights,
            msg_class_map=msg_class_map,
            rng=child_rng,
        ))

    # This generator must yield at least once to be a valid SimPy process
    yield env.timeout(0)


def _robot_send_loop(
    env: simpy.Environment,
    stack: MeshStack,
    tracker: Tracker,
    robot_id: int,
    bs_id: int,
    interval_s: float,
    class_names: list[str],
    class_weights: list[float],
    msg_class_map: dict[str, MessageClass],
    rng: np.random.Generator,
):
    """One robot's independent send loop. Offsets from other robots by a
    random phase so transmissions don't collide systematically."""
    seq = 0

    # Random initial offset within one interval — spreads robots in time
    offset = float(rng.uniform(0, interval_s))
    yield env.timeout(offset)

    while True:
        # Pick message class
        cls_name = rng.choice(class_names, p=class_weights)
        msg_cls = msg_class_map[cls_name]

        payload = f"{robot_id}:{seq}".encode()
        tracker.record_send(robot_id, seq, bs_id, env.now, msg_cls)

        stack.send(
            dst_addr=bs_id,
            message_class=msg_cls,
            payload=payload,
        )

        seq = (seq + 1) & 0xFFFF

        # Wait for next send — add small jitter to prevent re-synchronization
        jitter = float(rng.uniform(-0.05 * interval_s, 0.05 * interval_s))
        yield env.timeout(max(interval_s + jitter, 0.01))

# =====================================================================
# BS RX sniffer — catches ALL frames arriving at BS for PDR tracking
# =====================================================================

def make_bs_rx_sniffer(tracker: Tracker, env: simpy.Environment):
    """Returns an on_receive callback for the BS transport.

    This fires for EVERY frame the BS radio receives, before routing
    decides DELIVER/FORWARD/DROP. This is how we track PDR for flood
    (where routing never DELIVERs because dst=0xFFFF).
    """
    from titan_communication.radio.transport import ReceptionInfo
    from titan_communication.mesh.frame import Frame as _Frame

    def _sniffer(data: bytes, info: ReceptionInfo) -> None:
        try:
            frame = _Frame.unpack(data)
            tracker.record_rx_at_bs(frame, float(env.now))
        except Exception:
            pass  # corrupt frame, ignore

    return _sniffer


# =====================================================================
# Single run
# =====================================================================

def run_single(
    scheme_cfg: dict,
    cell: SweepCell,
    seed: int,
    cfg: dict,
) -> dict:
    """Run one simulation and return a metrics dict."""
    wall_start = time.monotonic()

    sim_duration = float(cfg["sim_duration_s"])
    warmup = float(cfg.get("warmup_s", 10.0))
    tick_interval = float(cfg.get("tick_interval_s", 1.0))

    rng = np.random.default_rng(seed)
    env = simpy.Environment()

    # ── LoRa params ──
    lora_params = make_lora_params(cfg)
    tx_power = float(cfg["lora"]["tx_power_dbm"])

    # ── Channel model (with extra obstruction) ──
    channel_model = make_channel_model(cfg, cell.extra_obstruction_db)
    virt_channel = VirtualChannel(env, channel_model, rng)

    # ── Duty policy ──
    duty_policy = make_duty_policy(cfg)

    # ── MAC channel-access backoff ──
    backoff_max_s = float(cfg.get("mac", {}).get("backoff_max_s", 0.0))

    # ── Topology (spacing is swept — see cell.spacing_m) ──
    topo = Topology.linear_chain(
        n_nodes=cell.n_nodes,
        spacing_m=float(cell.spacing_m),
    )

    # ── Tracker ──
    tracker = Tracker(warmup_s=warmup)

    # ── Create nodes ──
    bs_addr = 0x0001
    # Map topology node_id (0,1,2...) to mesh addresses (1,2,3...)
    # Node 0 in topology = BS = mesh addr 0x0001
    # Node k in topology = mesh addr k+1
    def mesh_addr(topo_id: int) -> int:
        return topo_id + 1

    mobile_positions: dict[int, MobilePosition] = {}
    stacks: dict[int, MeshStack] = {}
    robot_mobile_list: list[MobilePosition] = []

    for topo_id in topo.node_ids:
        addr = mesh_addr(topo_id)
        is_bs = (topo_id == topo.bs_id)
        x, y = topo.pos(topo_id)
        pos = MobilePosition(x, y)
        mobile_positions[addr] = pos

        if not is_bs:
            robot_mobile_list.append(pos)

        # Transport gets the real duty bucket
        transport_duty = DutyBucket(duty_policy)
        transport = VirtualLoRaTransport(
            env=env,
            channel=virt_channel,
            node_id=f"node_{addr:#06x}",
            params=lora_params,
            duty=transport_duty,
            position_getter=pos,
            tx_power_dbm=tx_power,
        )

        # BS sniffer — register BEFORE the stack's own on_receive
        if is_bs:
            sniffer = make_bs_rx_sniffer(tracker, env)
            transport.on_receive(sniffer)

        # Queue gets a no-cap duty bucket (pure priority ordering)
        queue_duty = DutyBucket(RegionPolicy.us915_fcc_raw())
        queue = PriorityQueue(env, queue_duty)

        # Routing
        is_sink = is_bs and cell.is_sink_only
        routing = make_routing(scheme_cfg, addr, bs_addr, is_bs=is_bs)

        # Stack — each node gets its own child RNG (derived from the run
        # seed) so the backoff draw is deterministic and reproducible.
        node_rng = np.random.default_rng(rng.integers(0, 2**31))
        stack = MeshStack(
            env=env,
            transport=transport,
            routing=routing,
            queue=queue,
            lora_params=lora_params,
            node_addr=addr,
            bs_addr=bs_addr,
            is_bs=is_bs,
            tick_interval_s=tick_interval,
            backoff_max_s=backoff_max_s,
            rng=node_rng,
        )
        stacks[addr] = stack

    # ── Start all stacks ──
    for stack in stacks.values():
        stack.start()

    # ── Mobility ──
    if cell.mobility_m_per_s > 0 and robot_mobile_list:
        area_half = topo.max_distance_to_bs() + 100.0
        env.process(mobility_process(
            env, robot_mobile_list, cell.mobility_m_per_s,
            area_half, rng,
        ))

    # ── Traffic ──
    env.process(traffic_generator(
        env=env,
        stacks=stacks,
        tracker=tracker,
        bs_id=bs_addr,
        offered_load_pkts_per_min_per_node=cell.offered_load_pkts_per_min_per_node,
        traffic_classes=cell.traffic_classes,
        rng=rng,
    ))

    # ── Run ──
    env.run(until=sim_duration)

    # ── Collect metrics ──
    tracker_metrics = tracker.compute_metrics()

    # Aggregate stack metrics
    tx_total = sum(s.tx_count for s in stacks.values())
    rx_total = sum(s.rx_count for s in stacks.values())
    deliver_total = sum(s.deliver_count for s in stacks.values())
    drop_total = sum(s.drop_count for s in stacks.values())
    forward_total = sum(s.forward_count for s in stacks.values())
    busy_wait_total = sum(s.busy_wait_count for s in stacks.values())

    # Control overhead: count CTRL frames in all logs / total data frames
    ctrl_tx_count = 0
    data_tx_count = 0
    for s in stacks.values():
        for entry in s.log:
            if entry.get("event") == "tx":
                if entry.get("class") == "CTRL":
                    ctrl_tx_count += 1
                else:
                    data_tx_count += 1
    control_overhead = (
        ctrl_tx_count / data_tx_count if data_tx_count > 0 else 0.0
    )

    # Duty utilization: mean across all robot transports at end of sim
    duty_utils = []
    for topo_id in topo.robot_ids:
        addr = mesh_addr(topo_id)
        transport = stacks[addr].transport
        util = transport.duty.utilization(sim_duration)
        duty_utils.append(util)
    duty_util_mean = sum(duty_utils) / len(duty_utils) if duty_utils else 0.0

    wall_s = time.monotonic() - wall_start

    return {
        "scheme": scheme_cfg["name"],
        "seed": seed,
        "n_nodes": cell.n_nodes,
        "traffic_mix": cell.traffic_mix_name,
        "mobility": cell.mobility_m_per_s,
        "obstruction": cell.extra_obstruction_db,
        "load": cell.offered_load_pkts_per_min_per_node,
        "bs_role": cell.bs_role_name,
        "spacing_m": cell.spacing_m,
        "pdr": tracker_metrics["pdr"],
        "latency_p50_ms": tracker_metrics["latency_p50_ms"],
        "latency_p95_ms": tracker_metrics["latency_p95_ms"],
        "control_overhead_ratio": round(control_overhead, 6),
        "duty_utilization_mean": round(duty_util_mean, 6),
        "tx_total": tx_total,
        "rx_total": rx_total,
        "deliver_total": deliver_total,
        "drop_total": drop_total,
        "forward_total": forward_total,
        "busy_wait_total": busy_wait_total,
        "wall_clock_s": round(wall_s, 3),
    }


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Titan DMS routing bake-off")
    parser.add_argument(
        "--config", type=str,
        default="src/titan_communication/bakeoff/scenarios.yaml",
        help="Path to scenarios.yaml",
    )
    parser.add_argument(
        "--out", type=str,
        default="src/titan_communication/bakeoff/results/",
        help="Output directory for results CSV",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print run count and exit without running",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cells = generate_sweep(cfg)
    schemes = cfg["schemes"]
    seed_start = int(cfg["seeds"]["start"])
    seed_count = int(cfg["seeds"]["count"])
    seeds = list(range(seed_start, seed_start + seed_count))

    total_runs = len(cells) * len(schemes) * len(seeds)
    print(f"Sweep: {len(cells)} cells × {len(schemes)} schemes × {len(seeds)} seeds = {total_runs} runs")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / cfg["output"]["results_file"]

    if args.dry_run:
        print(f"Output: {csv_path}")
        print("Dry run — exiting.")
        return

    # ── Resume support ──
    completed = load_completed(csv_path)
    print(f"Already completed: {len(completed)} runs")

    # ── Open CSV for append ──
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    # ── Run loop ──
    run_idx = 0
    skipped = 0
    failed = 0
    t_start = time.monotonic()

    try:
        for scheme_cfg in schemes:
            for cell in cells:
                for seed in seeds:
                    run_idx += 1
                    key = _run_key(scheme_cfg["name"], seed, cell)

                    if key in completed:
                        skipped += 1
                        continue

                    # Progress
                    elapsed = time.monotonic() - t_start
                    done_so_far = run_idx - skipped - failed
                    if done_so_far > 0:
                        rate = elapsed / done_so_far
                        remaining = (total_runs - run_idx) * rate
                        eta_min = remaining / 60.0
                    else:
                        eta_min = 0.0

                    print(
                        f"[{run_idx}/{total_runs}] "
                        f"{scheme_cfg['name']:8s} | "
                        f"n={cell.n_nodes} mix={cell.traffic_mix_name:10s} "
                        f"mob={cell.mobility_m_per_s} obs={cell.extra_obstruction_db:2.0f} "
                        f"load={cell.offered_load_pkts_per_min_per_node:3.0f} "
                        f"bs={cell.bs_role_name:18s} "
                        f"spacing={cell.spacing_m:.0f}m "
                        f"seed={seed:2d} | "
                        f"ETA {eta_min:.1f}min",
                        end="",
                        flush=True,
                    )

                    try:
                        result = run_single(scheme_cfg, cell, seed, cfg)
                        writer.writerow(result)
                        csv_file.flush()
                        print(
                            f" -> PDR={result['pdr']:.3f} "
                            f"lat={result['latency_p50_ms']:.1f}ms "
                            f"[{result['wall_clock_s']:.1f}s]"
                        )
                    except Exception as e:
                        failed += 1
                        print(f" -> FAILED: {e}")
                        continue

    except KeyboardInterrupt:
        print(f"\n\nInterrupted. {run_idx - skipped - failed} runs completed this session.")
        print(f"Resume by running the same command again.")
    finally:
        csv_file.close()

    elapsed_total = time.monotonic() - t_start
    print(f"\nDone. Total wall time: {elapsed_total / 60:.1f} min")
    print(f"  Completed: {run_idx - skipped - failed}")
    print(f"  Skipped (resume): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Results: {csv_path}")


if __name__ == "__main__":
    main()