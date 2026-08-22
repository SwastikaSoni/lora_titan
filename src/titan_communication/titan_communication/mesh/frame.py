"""Mesh frame format: 24 B binary header + opaque payload + 2 B CRC trailer.

Frame layout on-wire (26 + N bytes total):

    Header (24 B, big-endian):
        off  0  1B  version_and_flags  (v[4]|frag[1]|last[1]|ack[1]|reserved[1])
        off  1  1B  ttl                 (decremented per hop, 0 -> drop)
        off  2  1B  hop_count           (incremented per hop, metrics)
        off  3  1B  message_class       (0=SOS ... 4=BULK)
        off  4  2B  src_id
        off  6  2B  dst_id              (0xFFFF = broadcast)
        off  8  2B  prev_hop_id         (who last relayed us)
        off 10  2B  seq                 (per-src monotonic)
        off 12  1B  fragment_index      (0..fragment_total-1)
        off 13  1B  fragment_total      (>=1; 1 means not fragmented)
        off 14  8B  timestamp_ms        (sender's clock ms; interpretation
                                          is up to the mesh stack — sim
                                          time in Tier 1, Unix ms on hardware)
        off 22  2B  payload_len         (bytes in the payload segment)

    Payload (N bytes, 0..229):
        Opaque. The mesh layer conventionally msgpack-encodes structured
        data here (see encode_payload/decode_payload), but the frame
        itself does not care.

    Trailer (2 B):
        off 24+N  2B  crc16_ccitt (over header || payload)

Rationale for header layer:
    - CRC as a trailer (not inside the header) avoids the "zero the
      field, compute, write back" chicken-and-egg pattern.
    - 8-byte timestamp is generous but future-proof: 32-bit ms-since-epoch
      overflows in 2038, and sim-time drift over multi-day soak tests
      warrants headroom.
    - The mesh-layer CRC is redundant with the LoRa PHY CRC on-air, but
      catches software errors (buffer misuse, off-by-one during relay)
      that PHY CRC can't. Matches the paper's Algorithm 2 which does
      both PHY CRC and an application-layer checksum.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Final

import msgpack

__all__ = [
    "BROADCAST_ID",
    "CRC_INIT",
    "CRC_SIZE",
    "CRC_TRAILER_SIZE",
    "CURRENT_VERSION",
    "HEADER_SIZE",
    "MAX_PAYLOAD_BYTES",
    "MAX_TOTAL_BYTES",
    "Frame",
    "FrameError",
    "MessageClass",
    "crc16_ccitt",
    "decode_payload",
    "encode_payload",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CURRENT_VERSION: Final[int] = 1
HEADER_SIZE: Final[int] = 24
CRC_SIZE: Final[int] = 2
CRC_TRAILER_SIZE: Final[int] = CRC_SIZE  # alias for readability
MAX_TOTAL_BYTES: Final[int] = 255  # SX1276 PHY payload limit
MAX_PAYLOAD_BYTES: Final[int] = MAX_TOTAL_BYTES - HEADER_SIZE - CRC_SIZE  # 229
BROADCAST_ID: Final[int] = 0xFFFF
CRC_INIT: Final[int] = 0xFFFF

_HEADER_STRUCT: Final[struct.Struct] = struct.Struct(">BBBBHHHHBBQH")
assert _HEADER_STRUCT.size == HEADER_SIZE, (
    f"header struct size {_HEADER_STRUCT.size} != declared HEADER_SIZE {HEADER_SIZE}"
)


# ---------------------------------------------------------------------------
# MessageClass
# ---------------------------------------------------------------------------
class MessageClass(IntEnum):
    """Priority classes for the mesh queue. Lower value = higher priority.

    Named after their traffic type in the paper:
        SOS   — victim distress (EMERG device, §III-B.1)
        CTRL  — BS <-> robot commands (RoboMaC, §III-B.2)
        TELEM — periodic telemetry
        AUDIO — voice audio bursts (RoboTalkie, §III-B.3)
        BULK  — file transfers, logs, opportunistic traffic
    """

    SOS = 0
    CTRL = 1
    TELEM = 2
    AUDIO = 3
    BULK = 4


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class FrameError(RuntimeError):
    """Raised on any Frame.unpack failure: CRC, length, or unknown class."""


# ---------------------------------------------------------------------------
# CRC-16-CCITT-FALSE
# ---------------------------------------------------------------------------
def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no input/output reflection.

    Reference: ITU-T V.42 Annex A.1. Canonical test vector:
        crc16_ccitt(b"123456789") == 0x29B1
        crc16_ccitt(b"")          == 0xFFFF   (init, no XOR-out)

    Bit-by-bit implementation for clarity — throughput is not a concern
    at mesh packet rates (tens per second).
    """
    crc = CRC_INIT
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:  # noqa: SIM108
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


# ---------------------------------------------------------------------------
# msgpack payload helpers
# ---------------------------------------------------------------------------
def encode_payload(obj: Any) -> bytes:
    """Encode a Python object as a msgpack byte string.

    Uses ``use_bin_type=True`` so bytes are distinguishable from strings
    on decode (msgpack 1.x default). Callers must ensure the encoded
    output fits within ``MAX_PAYLOAD_BYTES`` (229 B).
    """
    return bytes(msgpack.packb(obj, use_bin_type=True))


def decode_payload(payload: bytes) -> Any:
    """Decode a msgpack byte string back to a Python object.

    ``raw=False`` returns str for msgpack strings (not bytes).
    ``strict_map_key=False`` allows non-str dict keys (int keys are
    useful for compact enum-tagged payloads).
    """
    return msgpack.unpackb(payload, raw=False, strict_map_key=False)


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------
@dataclass(frozen=True, repr=False)
class Frame:
    """A single mesh frame: header fields + opaque payload.

    Frames are immutable — mesh relays produce a modified copy via
    ``dataclasses.replace(frame, ttl=..., hop_count=..., prev_hop_id=...)``
    rather than mutating in place.
    """

    version: int
    is_fragment: bool
    is_last_fragment: bool
    requires_ack: bool
    ttl: int
    hop_count: int
    message_class: MessageClass
    src_id: int
    dst_id: int
    prev_hop_id: int
    seq: int
    fragment_index: int
    fragment_total: int
    timestamp_ms: int
    payload: bytes

    # -- Validation -------------------------------------------------------

    def __post_init__(self) -> None:
        if not 0 <= self.version <= 0xF:
            raise ValueError(f"version must be 0..15, got {self.version}")
        if not 0 <= self.ttl <= 0xFF:
            raise ValueError(f"ttl must be 0..255, got {self.ttl}")
        if not 0 <= self.hop_count <= 0xFF:
            raise ValueError(f"hop_count must be 0..255, got {self.hop_count}")
        for name in ("src_id", "dst_id", "prev_hop_id"):
            v = getattr(self, name)
            if not 0 <= v <= 0xFFFF:
                raise ValueError(f"{name} must be 0..0xFFFF, got {v}")
        if not 0 <= self.seq <= 0xFFFF:
            raise ValueError(f"seq must be 0..0xFFFF, got {self.seq}")
        if not 0 <= self.fragment_index <= 0xFF:
            raise ValueError(
                f"fragment_index must be 0..255, got {self.fragment_index}"
            )
        if not 1 <= self.fragment_total <= 0xFF:
            raise ValueError(
                f"fragment_total must be 1..255, got {self.fragment_total}"
            )
        if self.fragment_index >= self.fragment_total:
            raise ValueError(
                f"fragment_index {self.fragment_index} must be < "
                f"fragment_total {self.fragment_total}"
            )
        if not 0 <= self.timestamp_ms <= 0xFFFF_FFFF_FFFF_FFFF:
            raise ValueError(f"timestamp_ms out of uint64 range: {self.timestamp_ms}")
        if len(self.payload) > MAX_PAYLOAD_BYTES:
            raise ValueError(
                f"payload {len(self.payload)} B exceeds "
                f"MAX_PAYLOAD_BYTES ({MAX_PAYLOAD_BYTES} B)"
            )

    # -- Convenience ------------------------------------------------------

    @classmethod
    def new(
        cls,
        src_id: int,
        dst_id: int,
        message_class: MessageClass,
        payload: bytes,
        *,
        seq: int = 0,
        ttl: int = 8,
        timestamp_ms: int = 0,
        prev_hop_id: int = 0,
        hop_count: int = 0,
        requires_ack: bool = False,
    ) -> Frame:
        """Build a Frame with default values for the rarely-set fields.

        Defaults:
            version           = CURRENT_VERSION
            is_fragment       = False
            is_last_fragment  = False
            fragment_index    = 0
            fragment_total    = 1
            ttl               = 8   (typical mesh depth)
        """
        return cls(
            version=CURRENT_VERSION,
            is_fragment=False,
            is_last_fragment=False,
            requires_ack=requires_ack,
            ttl=ttl,
            hop_count=hop_count,
            message_class=message_class,
            src_id=src_id,
            dst_id=dst_id,
            prev_hop_id=prev_hop_id,
            seq=seq,
            fragment_index=0,
            fragment_total=1,
            timestamp_ms=timestamp_ms,
            payload=payload,
        )

    def total_size(self) -> int:
        """On-wire size in bytes: header + payload + CRC."""
        return HEADER_SIZE + len(self.payload) + CRC_SIZE

    def is_broadcast(self) -> bool:
        return self.dst_id == BROADCAST_ID

    def __repr__(self) -> str:
        return (
            f"Frame(v{self.version} {self.message_class.name} "
            f"{self.src_id}->{self.dst_id} seq={self.seq} "
            f"ttl={self.ttl} hop={self.hop_count} "
            f"pay={len(self.payload)}B"
            f"{' FRAG ' + str(self.fragment_index) + '/' + str(self.fragment_total) if self.is_fragment else ''}"
            f"{' ACK' if self.requires_ack else ''})"
        )

    # -- Pack / Unpack ----------------------------------------------------

    def pack(self) -> bytes:
        """Serialize to on-wire bytes: header || payload || crc trailer."""
        flags = self._pack_flags()
        header = _HEADER_STRUCT.pack(
            flags,
            self.ttl,
            self.hop_count,
            int(self.message_class),
            self.src_id,
            self.dst_id,
            self.prev_hop_id,
            self.seq,
            self.fragment_index,
            self.fragment_total,
            self.timestamp_ms,
            len(self.payload),
        )
        body = header + self.payload
        crc = crc16_ccitt(body)
        return body + struct.pack(">H", crc)

    @classmethod
    def unpack(cls, data: bytes) -> Frame:
        """Deserialize on-wire bytes back to a Frame.

        Raises:
            FrameError: too short, length mismatch, CRC mismatch, or
                        unknown message_class.
        """
        min_size = HEADER_SIZE + CRC_SIZE
        if len(data) < min_size:
            raise FrameError(
                f"frame too short: {len(data)} B < {min_size} B minimum"
            )

        header = data[:HEADER_SIZE]
        (
            flags_byte,
            ttl,
            hop_count,
            msg_class_int,
            src_id,
            dst_id,
            prev_hop_id,
            seq,
            frag_index,
            frag_total,
            timestamp_ms,
            payload_len,
        ) = _HEADER_STRUCT.unpack(header)

        expected_total = HEADER_SIZE + payload_len + CRC_SIZE
        if len(data) != expected_total:
            raise FrameError(
                f"frame length mismatch: got {len(data)} B, "
                f"header declares payload_len={payload_len} -> expected {expected_total} B"
            )

        payload = data[HEADER_SIZE : HEADER_SIZE + payload_len]
        (received_crc,) = struct.unpack(
            ">H", data[HEADER_SIZE + payload_len :]
        )
        computed_crc = crc16_ccitt(header + payload)
        if received_crc != computed_crc:
            raise FrameError(
                f"CRC mismatch: received 0x{received_crc:04X}, "
                f"computed 0x{computed_crc:04X}"
            )

        try:
            msg_class = MessageClass(msg_class_int)
        except ValueError as e:
            raise FrameError(
                f"unknown message_class {msg_class_int} "
                f"(valid: {sorted(int(c) for c in MessageClass)})"
            ) from e

        version, is_frag, is_last, req_ack = cls._unpack_flags(flags_byte)

        return cls(
            version=version,
            is_fragment=is_frag,
            is_last_fragment=is_last,
            requires_ack=req_ack,
            ttl=ttl,
            hop_count=hop_count,
            message_class=msg_class,
            src_id=src_id,
            dst_id=dst_id,
            prev_hop_id=prev_hop_id,
            seq=seq,
            fragment_index=frag_index,
            fragment_total=frag_total,
            timestamp_ms=timestamp_ms,
            payload=bytes(payload),
        )

    # -- Flag byte helpers (private) -------------------------------------

    def _pack_flags(self) -> int:
        return (
            ((self.version & 0x0F) << 4)
            | (0b1000 if self.is_fragment else 0)
            | (0b0100 if self.is_last_fragment else 0)
            | (0b0010 if self.requires_ack else 0)
        )

    @staticmethod
    def _unpack_flags(byte: int) -> tuple[int, bool, bool, bool]:
        version = (byte >> 4) & 0x0F
        is_fragment = bool(byte & 0b1000)
        is_last_fragment = bool(byte & 0b0100)
        requires_ack = bool(byte & 0b0010)
        return version, is_fragment, is_last_fragment, requires_ack
