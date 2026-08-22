# titan_ws — Enhanced Titan DMS workspace

This is the ROS 2 / SimPy workspace for the M.Tech project extending
Manuel et al. (IEEE IoT Journal, Oct 2024). See the project-root
`README.md` (the one attached to this repo/Claude project) for the full
plan, paper baseline, and Week-by-Week milestones.

## Current state

**Week 1–2 scaffolding.** Tooling only — no algorithms yet.

- `radio/` and `mesh/` packages exist but are empty (populated Weeks 1–2
  proper).
- `bakeoff/` package exists but is empty (populated Weeks 3–4).
- ROS 2 metadata (`package.xml`, `setup.py`, `resource/`) is **not**
  present yet. Added in Week 5 when we start wrapping nodes in `rclpy`.
  Rationale: keeps the Tier-1 SimPy code runnable without a `colcon
  build` cycle, and structurally enforces the "`radio/` and `mesh/`
  never import `rclpy`" rule from the project-root README §3.

## Setup (one-time)

```bash
cd ~/titan_ws
python3.12 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

`--system-site-packages` matters later when we source ROS 2 (Lyrical
ships its own Python packages we don't want to duplicate). For pure
Tier-1 work it's a no-op.

## Every-terminal setup (once ROS 2 is in the picture, Week 5+)

```bash
source /opt/ros/lyrical/setup.bash
source ~/titan_ws/install/setup.bash      # after first colcon build
source ~/titan_ws/.venv/bin/activate
```

For Weeks 1–2, only the venv line matters.

## Green-check commands

Run these to confirm the scaffolding is intact:

```bash
cd ~/titan_ws
source .venv/bin/activate
pytest              # unit tests
mypy                # type check
ruff check .        # lint
```

All three should exit 0 on a fresh clone.
