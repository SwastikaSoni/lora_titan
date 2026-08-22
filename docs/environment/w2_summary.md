# Weeks 1–2 summary — Tier-1 SimPy stack

## Milestone (from project-root README §5)

> pytest all-green on `radio/` and `mesh/`. Two SimPy processes exchange a
> frame with correct airtime, correct duty accounting, correct CRC.

Achieved.

## What shipped

| Module | Purpose | Tests | LoC (approx) |
|---|---|---|---|
| `radio/airtime.py` | Semtech SX1276 time-on-air formula, LDRO auto-rule, YAML preset loader | 46 | 210 |
| `radio/duty.py` | Duty-cycle + dwell-time bucket, 5 region policies (US915-polite/raw/hopping, IN865, EU868) | 39 | 240 |
| `radio/channel.py` | Log-distance pathloss + log-normal shadowing, sensitivity math, SX1276 demod SNR floors | 62 | 270 |
| `radio/transport.py` | `ILoRaTransport` + `IChannel` ABCs, `ReceptionInfo` value type | 24 | 200 |
| `radio/virtual.py` | SimPy `VirtualLoRaTransport` + `VirtualChannel` with collision model | 7 (smoke) | 320 |
| `mesh/frame.py` | 24 B binary header + CRC-16-CCITT + msgpack payload helpers | 70 | 350 |
| `mesh/queue.py` | Strict-priority 5-class queue, head-of-line blocking, duty-aware dequeue | 20 | 210 |

Total: ~1800 LoC production, ~2400 LoC test, **310 tests passing**.

## Design decisions locked in

- **US915-polite = 10% self-imposed duty cap** (FCC §15.247 has no cap;
  we impose one to make the mesh's deferral algorithm testable). See
  `radio/duty.py` docstring for the reasoning paragraph that will end up
  in the thesis methodology chapter.
- **PAPER_DEFAULT** = SF=10 / BW=500 kHz / CR=4/5 / preamble=8 / explicit
  header / CRC on / TX=15 dBm. Matches Manuel et al. §III-B, Algorithm 2.
  Time-on-air for a 32-byte payload: **113.15 ms**.
- **Mesh header = 24 B** exactly, CRC = 2 B trailer, max payload = 229 B.
- **CRC-16-CCITT-FALSE** (poly 0x1021, init 0xFFFF). Verified against the
  canonical `crc16("123456789") == 0x29B1` test vector.
- **msgpack** for payload encoding (over JSON/protobuf).
- **Strict priority + head-of-line blocking** in the queue. No skip-past.
  If SOS is duty-blocked, no lower-class frame goes out.
- **No in-flight preemption**. LoRa PHY can't stop mid-chirp.
- **Collision model**: all-or-nothing, no capture effect. Pessimistic
  bias, appropriate for SAR-critical protocol.
- **SF/BW/CR mismatch = silent drop**. Matches real SX1276 behaviour.

## Known limitations

Carried forward — not blockers, tracked for future weeks:

1. **Shadowing is per-packet-independent.** Real shadowing has ~10 m
   spatial decorrelation. Fine for the routing bake-off; revisit if
   adaptive leader recovery shows RSSI-trend noise dominates.
2. **No capture effect.** Two overlapping packets always both drop. Real
   LoRa can recover the stronger one with ~6 dB SNR margin. Refinement
   deferred to post-bake-off.
3. **Cryptography (Manuel & Daimi [19]) not integrated.** README §7
   explicit deferral — future work.
4. **No LoRaWAN comparison.** Out of scope per README §7.
5. **`wake_duty_waiters` is a no-op placeholder** in the queue. Kept in
   the interface; mesh stack code has a hook if we need explicit
   duty-state signalling later.

## Deferred to future steps

- **`radio/channel.py` open question**: which Petäjäjärvi 2015 constants
  best match Gazebo obstruction? Sensitivity study is Week 4.
- **Routing scheme**: decided by Week 4 bake-off. `mesh/routing_*.py`
  files don't exist yet.
- **BS as mesh node vs designated sink**: bake-off decides.

## What Week 3 needs from Weeks 1–2

The routing bake-off in Weeks 3–4 will use:
- `time_on_air_s(params, len(frame.pack()))` to size airtime per frame
- `DutyBucket` per node to enforce policy
- `ChannelModel` swapped across obstruction levels (10/20 dB extra loss)
- `Frame` + `MessageClass` to generate mixed traffic
- `PriorityQueue` per node for outgoing frames
- `VirtualLoRaTransport` + `VirtualChannel` as the substrate

All the pieces exist and compose. Week 3 writes the routing schemes on
top.