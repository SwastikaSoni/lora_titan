

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Topology:
    """A set of named node positions with distance helpers."""

    positions: dict[int, tuple[float, float]]
    bs_id: int = 0
    layout_name: str = ""

    @property
    def n_nodes(self) -> int:
        return len(self.positions)

    @property
    def node_ids(self) -> list[int]:
        return sorted(self.positions.keys())

    @property
    def robot_ids(self) -> list[int]:
        """All node IDs except the BS."""
        return [nid for nid in self.node_ids if nid != self.bs_id]

    def pos(self, node_id: int) -> tuple[float, float]:
        return self.positions[node_id]

    def distance(self, a: int, b: int) -> float:
        """Euclidean distance between two nodes in metres."""
        pa, pb = self.positions[a], self.positions[b]
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

    def distance_matrix(self) -> dict[tuple[int, int], float]:
        """All pairwise distances. Symmetric: (a,b) and (b,a) both present."""
        ids = self.node_ids
        out: dict[tuple[int, int], float] = {}
        for i, a in enumerate(ids):
            for b in ids[i:]:
                d = self.distance(a, b)
                out[(a, b)] = d
                out[(b, a)] = d
        return out

    def max_distance_to_bs(self) -> float:
        """Farthest robot from BS."""
        return max(self.distance(self.bs_id, r) for r in self.robot_ids)

    def summary(self) -> str:
        """Human-readable one-liner for logs."""
        return (
            f"Topology({self.layout_name}, n={self.n_nodes}, "
            f"max_bs_dist={self.max_distance_to_bs():.1f}m)"
        )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def linear_chain(
        cls,
        n_nodes: int,
        spacing_m: float,
        bs_id: int = 0,
    ) -> Topology:
        """Nodes in a line: BS — R1 — R2 — ... — R(n-1).

        Good for testing multi-hop depth. Max hops = n_nodes - 1.

        Args:
            n_nodes:   total nodes including BS.
            spacing_m: distance between consecutive nodes.
        """
        _validate(n_nodes, 2)
        positions = {i: (i * spacing_m, 0.0) for i in range(n_nodes)}
        return cls(positions=positions, bs_id=bs_id, layout_name="linear_chain")

    @classmethod
    def grid(
        cls,
        n_nodes: int,
        spacing_m: float,
        bs_id: int = 0,
    ) -> Topology:
        """Nodes on a rectangular grid, BS at position 0 (top-left corner).

        Grid is as square as possible: cols = ceil(sqrt(n)), rows = ceil(n/cols).

        Args:
            n_nodes:   total nodes including BS.
            spacing_m: distance between adjacent grid cells.
        """
        _validate(n_nodes, 2)
        cols = math.ceil(math.sqrt(n_nodes))
        positions: dict[int, tuple[float, float]] = {}
        for i in range(n_nodes):
            row = i // cols
            col = i % cols
            positions[i] = (col * spacing_m, row * spacing_m)
        return cls(positions=positions, bs_id=bs_id, layout_name="grid")

    @classmethod
    def star(
        cls,
        n_nodes: int,
        radius_m: float,
        bs_id: int = 0,
    ) -> Topology:
        """BS at centre, robots evenly spaced on a circle of radius_m.

        All robots are 1-hop from BS (if radius_m < LoRa range).
        Tests load balancing and contention, not multi-hop.

        Args:
            n_nodes:   total nodes including BS.
            radius_m:  ring radius.
        """
        _validate(n_nodes, 2)
        positions: dict[int, tuple[float, float]] = {bs_id: (0.0, 0.0)}
        n_robots = n_nodes - 1
        for i in range(n_robots):
            angle = 2.0 * math.pi * i / n_robots
            nid = i + 1 if i < bs_id else i + 1  # skip bs_id
            # Simpler: just use sequential ids, bs_id is always 0
            positions[i + 1] = (
                radius_m * math.cos(angle),
                radius_m * math.sin(angle),
            )
        return cls(positions=positions, bs_id=bs_id, layout_name="star")

    @classmethod
    def random_area(
        cls,
        n_nodes: int,
        width_m: float,
        height_m: float,
        rng: np.random.Generator,
        bs_id: int = 0,
        bs_position: tuple[float, float] = (0.0, 0.0),
        min_separation_m: float = 5.0,
    ) -> Topology:
        """BS at a fixed position, robots uniformly scattered in a rectangle.

        The rectangle is centred on bs_position. Robots are re-sampled
        if they land within min_separation_m of any existing node
        (prevents degenerate co-located nodes).

        Args:
            n_nodes:           total nodes including BS.
            width_m, height_m: area dimensions.
            rng:               numpy Generator for reproducibility.
            bs_position:       BS coordinates.
            min_separation_m:  minimum inter-node distance.
        """
        _validate(n_nodes, 2)
        positions: dict[int, tuple[float, float]] = {bs_id: bs_position}

        x_min = bs_position[0] - width_m / 2
        y_min = bs_position[1] - height_m / 2

        max_attempts = 1000
        nid = 1
        while nid < n_nodes:
            for _ in range(max_attempts):
                x = float(rng.uniform(x_min, x_min + width_m))
                y = float(rng.uniform(y_min, y_min + height_m))
                # Check separation
                ok = all(
                    math.hypot(x - px, y - py) >= min_separation_m
                    for px, py in positions.values()
                )
                if ok:
                    positions[nid] = (x, y)
                    nid += 1
                    break
            else:
                raise RuntimeError(
                    f"Could not place node {nid} after {max_attempts} attempts. "
                    f"Area too small or min_separation_m too large."
                )

        return cls(positions=positions, bs_id=bs_id, layout_name="random_area")

    @classmethod
    def cluster(
        cls,
        n_nodes: int,
        cluster_distances_m: list[float],
        cluster_radius_m: float,
        rng: np.random.Generator,
        bs_id: int = 0,
    ) -> Topology:
        """BS at centre, robots grouped in K clusters at various distances.

        Mimics the paper's deployment: some robots near BS (Region 2),
        some far (Region 3). Each cluster is a tight group of nodes
        scattered within cluster_radius_m of the cluster centre.

        Cluster centres are placed at cluster_distances_m from BS,
        spread evenly around the compass. Robots are divided as equally
        as possible among clusters.

        Args:
            n_nodes:              total nodes including BS.
            cluster_distances_m:  list of distances from BS, one per cluster.
            cluster_radius_m:     scatter radius within each cluster.
            rng:                  numpy Generator.
        """
        _validate(n_nodes, 2)
        n_robots = n_nodes - 1
        k = len(cluster_distances_m)
        if k == 0:
            raise ValueError("Need at least one cluster distance")

        # Divide robots among clusters
        base_per_cluster = n_robots // k
        remainder = n_robots % k
        cluster_sizes = [base_per_cluster + (1 if i < remainder else 0) for i in range(k)]

        positions: dict[int, tuple[float, float]] = {bs_id: (0.0, 0.0)}
        nid = 1
        for ci, (dist, size) in enumerate(zip(cluster_distances_m, cluster_sizes)):
            # Cluster centre: evenly spaced angle from BS
            angle = 2.0 * math.pi * ci / k
            cx = dist * math.cos(angle)
            cy = dist * math.sin(angle)

            for _ in range(size):
                # Scatter within cluster_radius_m of centre
                r = float(rng.uniform(0, cluster_radius_m))
                a = float(rng.uniform(0, 2.0 * math.pi))
                positions[nid] = (cx + r * math.cos(a), cy + r * math.sin(a))
                nid += 1

        return cls(positions=positions, bs_id=bs_id, layout_name="cluster")

    # ------------------------------------------------------------------
    # Mobility helper
    # ------------------------------------------------------------------

    def with_noise(
        self,
        rng: np.random.Generator,
        jitter_m: float = 5.0,
    ) -> Topology:
        """Return a new Topology with random positional jitter added.

        BS position is NOT jittered (it's fixed infrastructure).
        Useful for generating multiple slightly-different instances
        from the same base layout across seeds.
        """
        new_pos: dict[int, tuple[float, float]] = {}
        for nid, (x, y) in self.positions.items():
            if nid == self.bs_id:
                new_pos[nid] = (x, y)
            else:
                dx = float(rng.uniform(-jitter_m, jitter_m))
                dy = float(rng.uniform(-jitter_m, jitter_m))
                new_pos[nid] = (x + dx, y + dy)
        return Topology(
            positions=new_pos,
            bs_id=self.bs_id,
            layout_name=self.layout_name + "+noise",
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(n_nodes: int, minimum: int = 2) -> None:
    if n_nodes < minimum:
        raise ValueError(f"n_nodes must be >= {minimum}, got {n_nodes}")