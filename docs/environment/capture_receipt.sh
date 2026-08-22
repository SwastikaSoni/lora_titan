#!/usr/bin/env bash
# Captures a reproducibility receipt for the current milestone.
# Usage: bash docs/environment/capture_receipt.sh w2
# Writes: docs/environment/receipt_<tag>.md
#         docs/environment/requirements-frozen-<tag>.txt

set -euo pipefail

TAG="${1:-adhoc}"
WS_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$WS_ROOT"

RECEIPT="docs/environment/receipt_${TAG}.md"
FROZEN="docs/environment/requirements-frozen-${TAG}.txt"

# Freeze exact package versions (pip hashes are separate — we don't need
# hash-pinning at this stage; add --require-hashes later if we ship).
./.venv/bin/pip freeze > "$FROZEN"

# Green-check counts (captured live so the receipt matches reality).
PYTEST_OUT="$(./.venv/bin/pytest -q --no-header 2>&1 | tail -1)"
MYPY_OUT="$(./.venv/bin/mypy 2>&1 | tail -1)"
RUFF_OUT="$(./.venv/bin/ruff check . 2>&1 | tail -1)"

# Git state — best-effort; skip if not a repo yet.
GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo')"
GIT_DIRTY=""
if git diff-index --quiet HEAD -- 2>/dev/null; then
    GIT_DIRTY="clean"
else
    GIT_DIRTY="dirty (uncommitted changes)"
fi

cat > "$RECEIPT" <<EOF
# Environment receipt — ${TAG}

Captured: $(date -Iseconds)
Host: $(hostname)

## System
- OS: $(lsb_release -ds 2>/dev/null || uname -sr)
- Kernel: $(uname -r)
- Arch: $(uname -m)
- Python: $(./.venv/bin/python --version)
- Venv path: ${WS_ROOT}/.venv

## Git
- Commit: ${GIT_SHA}
- Working tree: ${GIT_DIRTY}

## Green-check results
- pytest: ${PYTEST_OUT}
- mypy:   ${MYPY_OUT}
- ruff:   ${RUFF_OUT}

## Frozen dependency versions
See \`${FROZEN}\` (output of \`pip freeze\`).

## Reproduce
\`\`\`bash
cd ~/titan_ws
python3 -m venv .venv
./.venv/bin/pip install -r ${FROZEN}
./.venv/bin/pytest
\`\`\`
EOF

echo "wrote ${RECEIPT}"
echo "wrote ${FROZEN}"