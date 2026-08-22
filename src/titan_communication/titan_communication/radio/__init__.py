"""Radio layer — LoRa physical/link primitives.

Populated in Weeks 1–2:
    airtime.py    — Semtech time-on-air formula (SF, BW, CR, payload)
    duty.py       — per-node duty-cycle bucket
    channel.py    — log-distance pathloss + log-normal shadowing
    transport.py  — ILoRaTransport ABC (SimPy virtual + ESP32 SX1276)

Nothing here imports rclpy.
"""
