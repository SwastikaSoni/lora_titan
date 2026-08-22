"""Mesh layer — LoRa network primitives above the radio.

Populated in Weeks 1–2 (frame, queue) and Weeks 3–4 (routing, stack):
    frame.py            — 24 B header pack/unpack + CRC-16-CCITT
    queue.py            — 5-class priority queue with preemption
    routing_flood.py    — flooding with dedup cache
    routing_aodv.py     — AODV-lite (reactive)
    routing_gradient.py — proactive gradient to sink
    stack.py            — frame + queue + routing + transport composition

Nothing here imports rclpy.
"""
