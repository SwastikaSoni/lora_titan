"""Titan DMS communication stack — LoRa mesh + virtual radio.

Structure (README §3):
    radio/  — physical/link layer: airtime, channel, duty, transport
    mesh/   — network layer: frame, queue, routing, stack
    nodes/  — ROS 2 wrapper nodes (added Week 5+)

Invariant: nothing under ``radio/`` or ``mesh/`` imports ``rclpy``. This
keeps the code runnable in the Tier-1 SimPy harness (no ROS) and lets
the same stack later port to Arduino/ESP32 by swapping the transport.
"""

__version__ = "0.1.0"
