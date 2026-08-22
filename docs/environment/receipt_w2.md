# Environment receipt — w2

Captured: 2026-08-23T00:18:58+05:30
Host: hp-HP-Z2-Tower-G9-Workstation-Desktop-PC

## System
- OS: Ubuntu 26.04 LTS
- Kernel: 7.0.0-30-generic
- Arch: x86_64
- Python: Python 3.14.4
- Venv path: /home/hp/titan_ws/.venv

## Git
- Commit: HEAD
not-a-git-repo
- Working tree: dirty (uncommitted changes)

## Green-check results
- pytest: 310 passed in 0.22s
- mypy:   Success: no issues found in 10 source files
- ruff:   All checks passed!

## Frozen dependency versions
See `docs/environment/requirements-frozen-w2.txt` (output of `pip freeze`).

## Reproduce
```bash
cd ~/titan_ws
python3 -m venv .venv
./.venv/bin/pip install -r docs/environment/requirements-frozen-w2.txt
./.venv/bin/pytest
```
