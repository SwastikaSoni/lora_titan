"""Smoke test — proves the scaffolding is wired up.

If this file fails, nothing else will work. If it passes, we know:
    - pytest discovers tests under src/titan_communication/tests/
    - titan_communication is importable (pythonpath in pyproject.toml)
    - the Python version is what we expect
    - the runtime deps installed cleanly
"""

from __future__ import annotations

import sys


def test_package_importable() -> None:
    """The titan_communication package can be imported."""
    import titan_communication

    assert titan_communication.__version__ == "0.1.0"


def test_subpackages_importable() -> None:
    """The radio/ and mesh/ subpackages exist and import cleanly."""
    from titan_communication import mesh, radio  # noqa: F401


def test_bakeoff_importable() -> None:
    """bakeoff is a top-level peer package (README §3), not a subpackage."""
    import bakeoff  # noqa: F401


def test_python_version_supported() -> None:
    """We require Python 3.10+ (structural pattern matching, PEP 604 unions)."""
    assert sys.version_info >= (3, 10), (
        f"Need Python >= 3.10, got {sys.version_info[:3]}"
    )


def test_runtime_deps_importable() -> None:
    """SimPy, msgpack, numpy, yaml all install and import."""
    import msgpack  # noqa: F401
    import numpy  # noqa: F401
    import simpy  # noqa: F401
    import yaml  # noqa: F401
