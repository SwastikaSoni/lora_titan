"""Tests for the mesh frame: pack/unpack, CRC, validation, msgpack helpers."""

from __future__ import annotations

import dataclasses
import struct

import pytest
from titan_communication.mesh.frame import (
    BROADCAST_ID,
    CRC_SIZE,
    CURRENT_VERSION,
    HEADER_SIZE,
    MAX_PAYLOAD_BYTES,
    MAX_TOTAL_BYTES,
    Frame,
    FrameError,
    MessageClass,
    crc16_ccitt,
    decode_payload,
    encode_payload,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
class TestConstants:
    def test_header_size_is_24(self) -> None:
        assert HEADER_SIZE == 24

    def test_crc_size_is_2(self) -> None:
        assert CRC_SIZE == 2

    def test_max_total_matches_sx1276(self) -> None:
        assert MAX_TOTAL_BYTES == 255

    def test_max_payload_derived_correctly(self) -> None:
        assert MAX_PAYLOAD_BYTES == 255 - 24 - 2
        assert MAX_PAYLOAD_BYTES == 229

    def test_broadcast_id(self) -> None:
        assert BROADCAST_ID == 0xFFFF

    def test_current_version(self) -> None:
        assert CURRENT_VERSION == 1


# ---------------------------------------------------------------------------
# MessageClass
# ---------------------------------------------------------------------------
class TestMessageClass:
    def test_values_zero_through_four(self) -> None:
        assert int(MessageClass.SOS) == 0
        assert int(MessageClass.CTRL) == 1
        assert int(MessageClass.TELEM) == 2
        assert int(MessageClass.AUDIO) == 3
        assert int(MessageClass.BULK) == 4

    def test_lower_value_is_higher_priority(self) -> None:
        # This ordering matters for the priority queue in step 7.
        assert MessageClass.SOS < MessageClass.CTRL
        assert MessageClass.CTRL < MessageClass.TELEM
        assert MessageClass.TELEM < MessageClass.AUDIO
        assert MessageClass.AUDIO < MessageClass.BULK


# ---------------------------------------------------------------------------
# CRC-16-CCITT-FALSE
# ---------------------------------------------------------------------------
class TestCRC16CCITT:
    def test_canonical_test_vector(self) -> None:
        # Every CRC-16/CCITT-FALSE reference lists this one.
        assert crc16_ccitt(b"123456789") == 0x29B1

    def test_empty_input_returns_init(self) -> None:
        # No XOR-out step; init value passes through.
        assert crc16_ccitt(b"") == 0xFFFF

    def test_deterministic(self) -> None:
        data = b"the quick brown fox jumps over the lazy dog"
        assert crc16_ccitt(data) == crc16_ccitt(data)

    def test_single_byte_change_changes_crc(self) -> None:
        base = b"the quick brown fox"
        flipped = b"tHe quick brown fox"
        assert crc16_ccitt(base) != crc16_ccitt(flipped)

    def test_single_bit_flip_changes_crc(self) -> None:
        # Flip the low bit of the first byte.
        base = b"hello world"
        flipped = bytes([base[0] ^ 0x01]) + base[1:]
        assert crc16_ccitt(base) != crc16_ccitt(flipped)

    def test_returns_uint16_range(self) -> None:
        for data in [b"", b"x", b"x" * 100, b"\x00" * 50, b"\xff" * 50]:
            crc = crc16_ccitt(data)
            assert 0 <= crc <= 0xFFFF


# ---------------------------------------------------------------------------
# Frame validation
# ---------------------------------------------------------------------------
class TestFrameValidation:
    def _valid_kwargs(self) -> dict[str, object]:
        return dict(
            version=1,
            is_fragment=False,
            is_last_fragment=False,
            requires_ack=False,
            ttl=8,
            hop_count=0,
            message_class=MessageClass.CTRL,
            src_id=1,
            dst_id=2,
            prev_hop_id=1,
            seq=0,
            fragment_index=0,
            fragment_total=1,
            timestamp_ms=1000,
            payload=b"hello",
        )

    def test_valid_construction(self) -> None:
        f = Frame(**self._valid_kwargs())  # type: ignore[arg-type]
        assert f.src_id == 1
        assert f.payload == b"hello"

    @pytest.mark.parametrize("v", [-1, 16, 100])
    def test_invalid_version_raises(self, v: int) -> None:
        kw = self._valid_kwargs() | {"version": v}
        with pytest.raises(ValueError, match="version"):
            Frame(**kw)  # type: ignore[arg-type]

    @pytest.mark.parametrize("ttl", [-1, 256, 1000])
    def test_invalid_ttl_raises(self, ttl: int) -> None:
        kw = self._valid_kwargs() | {"ttl": ttl}
        with pytest.raises(ValueError, match="ttl"):
            Frame(**kw)  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["src_id", "dst_id", "prev_hop_id"])
    @pytest.mark.parametrize("v", [-1, 0x10000, 0xFFFFFF])
    def test_invalid_ids_raise(self, field: str, v: int) -> None:
        kw = self._valid_kwargs() | {field: v}
        with pytest.raises(ValueError, match=field):
            Frame(**kw)  # type: ignore[arg-type]

    def test_invalid_seq_raises(self) -> None:
        kw = self._valid_kwargs() | {"seq": 0x10000}
        with pytest.raises(ValueError, match="seq"):
            Frame(**kw)  # type: ignore[arg-type]

    @pytest.mark.parametrize("total", [0, -1, 256])
    def test_invalid_fragment_total_raises(self, total: int) -> None:
        kw = self._valid_kwargs() | {"fragment_total": total}
        with pytest.raises(ValueError, match="fragment_total"):
            Frame(**kw)  # type: ignore[arg-type]

    def test_fragment_index_ge_total_raises(self) -> None:
        kw = self._valid_kwargs() | {"fragment_index": 3, "fragment_total": 3}
        with pytest.raises(ValueError, match="fragment_index"):
            Frame(**kw)  # type: ignore[arg-type]

    def test_payload_too_large_raises(self) -> None:
        kw = self._valid_kwargs() | {"payload": b"x" * (MAX_PAYLOAD_BYTES + 1)}
        with pytest.raises(ValueError, match="MAX_PAYLOAD_BYTES"):
            Frame(**kw)  # type: ignore[arg-type]

    def test_payload_at_max_ok(self) -> None:
        kw = self._valid_kwargs() | {"payload": b"x" * MAX_PAYLOAD_BYTES}
        f = Frame(**kw)  # type: ignore[arg-type]
        assert len(f.payload) == MAX_PAYLOAD_BYTES

    def test_frame_is_frozen(self) -> None:
        f = Frame(**self._valid_kwargs())  # type: ignore[arg-type]
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.ttl = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Frame.new factory
# ---------------------------------------------------------------------------
class TestFrameNewFactory:
    def test_minimal_call(self) -> None:
        f = Frame.new(src_id=1, dst_id=2, message_class=MessageClass.CTRL, payload=b"x")
        assert f.version == CURRENT_VERSION
        assert f.is_fragment is False
        assert f.is_last_fragment is False
        assert f.requires_ack is False
        assert f.ttl == 8
        assert f.hop_count == 0
        assert f.fragment_index == 0
        assert f.fragment_total == 1

    def test_overrides(self) -> None:
        f = Frame.new(
            src_id=1,
            dst_id=2,
            message_class=MessageClass.SOS,
            payload=b"help",
            seq=42,
            ttl=4,
            timestamp_ms=12345,
            requires_ack=True,
        )
        assert f.seq == 42
        assert f.ttl == 4
        assert f.timestamp_ms == 12345
        assert f.requires_ack is True


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------
class TestFrameHelpers:
    def test_total_size(self) -> None:
        f = Frame.new(1, 2, MessageClass.CTRL, b"x" * 10)
        assert f.total_size() == HEADER_SIZE + 10 + CRC_SIZE
        assert f.total_size() == 36

    def test_is_broadcast_true(self) -> None:
        f = Frame.new(1, BROADCAST_ID, MessageClass.SOS, b"help")
        assert f.is_broadcast()

    def test_is_broadcast_false(self) -> None:
        f = Frame.new(1, 2, MessageClass.SOS, b"help")
        assert not f.is_broadcast()

    def test_repr_includes_key_fields(self) -> None:
        f = Frame.new(1, 2, MessageClass.SOS, b"help", seq=42)
        r = repr(f)
        assert "SOS" in r
        assert "1->2" in r
        assert "seq=42" in r

    def test_repr_does_not_dump_payload_bytes(self) -> None:
        # A 200 B payload's bytes shouldn't appear literally in the repr.
        f = Frame.new(1, 2, MessageClass.BULK, b"A" * 200)
        r = repr(f)
        assert "A" * 200 not in r
        assert "200B" in r


# ---------------------------------------------------------------------------
# Frame.pack — on-wire structure
# ---------------------------------------------------------------------------
class TestFramePackStructure:
    def test_total_length_matches_declared(self) -> None:
        f = Frame.new(1, 2, MessageClass.CTRL, b"hello")
        assert len(f.pack()) == f.total_size()

    def test_empty_payload_size(self) -> None:
        f = Frame.new(1, 2, MessageClass.CTRL, b"")
        assert len(f.pack()) == HEADER_SIZE + CRC_SIZE
        assert len(f.pack()) == 26

    def test_max_payload_fits_in_lora_phy(self) -> None:
        f = Frame.new(1, 2, MessageClass.BULK, b"x" * MAX_PAYLOAD_BYTES)
        assert len(f.pack()) == MAX_TOTAL_BYTES

    def test_first_byte_encodes_version(self) -> None:
        f = Frame.new(1, 2, MessageClass.CTRL, b"x")
        # version=1 -> high nibble = 0001, low nibble = 0000
        assert f.pack()[0] == 0x10

    def test_flag_bits_pack_correctly(self) -> None:
        # is_fragment=True (bit 3), last_fragment=True (bit 2), req_ack=True (bit 1)
        f = Frame(
            version=1,
            is_fragment=True,
            is_last_fragment=True,
            requires_ack=True,
            ttl=8,
            hop_count=0,
            message_class=MessageClass.CTRL,
            src_id=1,
            dst_id=2,
            prev_hop_id=1,
            seq=0,
            fragment_index=1,
            fragment_total=2,
            timestamp_ms=0,
            payload=b"x",
        )
        # version=1 (0x10) | frag (0x08) | last (0x04) | ack (0x02) = 0x1E
        assert f.pack()[0] == 0x1E

    def test_message_class_at_offset_3(self) -> None:
        f = Frame.new(1, 2, MessageClass.SOS, b"x")
        assert f.pack()[3] == 0

    def test_src_dst_at_offsets_4_and_6(self) -> None:
        f = Frame.new(0x1234, 0x5678, MessageClass.CTRL, b"x")
        packed = f.pack()
        (src,) = struct.unpack(">H", packed[4:6])
        (dst,) = struct.unpack(">H", packed[6:8])
        assert src == 0x1234
        assert dst == 0x5678

    def test_trailer_is_crc_of_header_and_payload(self) -> None:
        f = Frame.new(1, 2, MessageClass.CTRL, b"hello")
        packed = f.pack()
        body = packed[: HEADER_SIZE + len(f.payload)]
        (trailer_crc,) = struct.unpack(">H", packed[-CRC_SIZE:])
        assert trailer_crc == crc16_ccitt(body)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------
class TestFrameRoundTrip:
    @pytest.mark.parametrize(
        "msg_class",
        [MessageClass.SOS, MessageClass.CTRL, MessageClass.TELEM,
         MessageClass.AUDIO, MessageClass.BULK],
    )
    def test_round_trip_each_class(self, msg_class: MessageClass) -> None:
        f = Frame.new(src_id=42, dst_id=1000, message_class=msg_class, payload=b"data")
        assert Frame.unpack(f.pack()) == f

    @pytest.mark.parametrize(
        "payload",
        [b"", b"x", b"hello", b"A" * 100, b"A" * MAX_PAYLOAD_BYTES,
         bytes(range(256))[:MAX_PAYLOAD_BYTES]],
    )
    def test_round_trip_various_payloads(self, payload: bytes) -> None:
        f = Frame.new(1, 2, MessageClass.CTRL, payload)
        recovered = Frame.unpack(f.pack())
        assert recovered.payload == payload
        assert recovered == f

    def test_round_trip_with_all_flags(self) -> None:
        f = Frame(
            version=15,  # max version
            is_fragment=True,
            is_last_fragment=True,
            requires_ack=True,
            ttl=255,
            hop_count=42,
            message_class=MessageClass.AUDIO,
            src_id=0xABCD,
            dst_id=BROADCAST_ID,
            prev_hop_id=0x1234,
            seq=0xFFFF,
            fragment_index=7,
            fragment_total=8,
            timestamp_ms=(1 << 63),  # large but valid uint64
            payload=b"\x00\x01\x02\xff\xfe\xfd",
        )
        assert Frame.unpack(f.pack()) == f

    def test_round_trip_preserves_id_ranges(self) -> None:
        for src, dst in [(0, 0), (1, 2), (0xFFFF, 0xFFFE), (0, BROADCAST_ID)]:
            f = Frame.new(src, dst, MessageClass.CTRL, b"x")
            r = Frame.unpack(f.pack())
            assert r.src_id == src
            assert r.dst_id == dst


# ---------------------------------------------------------------------------
# Corruption detection
# ---------------------------------------------------------------------------
class TestFrameCorruptionDetection:
    def test_bit_flip_in_header_raises(self) -> None:
        f = Frame.new(1, 2, MessageClass.CTRL, b"hello")
        packed = bytearray(f.pack())
        packed[5] ^= 0x01  # flip a bit in src_id area
        with pytest.raises(FrameError, match="CRC mismatch"):
            Frame.unpack(bytes(packed))

    def test_bit_flip_in_payload_raises(self) -> None:
        f = Frame.new(1, 2, MessageClass.CTRL, b"hello")
        packed = bytearray(f.pack())
        packed[HEADER_SIZE] ^= 0x01  # flip a bit in payload
        with pytest.raises(FrameError, match="CRC mismatch"):
            Frame.unpack(bytes(packed))

    def test_bit_flip_in_crc_raises(self) -> None:
        f = Frame.new(1, 2, MessageClass.CTRL, b"hello")
        packed = bytearray(f.pack())
        packed[-1] ^= 0x01  # flip a bit in trailer CRC
        with pytest.raises(FrameError, match="CRC mismatch"):
            Frame.unpack(bytes(packed))

    def test_too_short_raises(self) -> None:
        with pytest.raises(FrameError, match="too short"):
            Frame.unpack(b"\x00" * 25)  # < HEADER_SIZE + CRC_SIZE

    def test_length_mismatch_raises(self) -> None:
        f = Frame.new(1, 2, MessageClass.CTRL, b"hello")
        packed = f.pack()
        # Truncate by one byte -> length no longer matches declared payload_len.
        with pytest.raises(FrameError, match="length mismatch"):
            Frame.unpack(packed[:-1])

    def test_unknown_message_class_raises(self) -> None:
        f = Frame.new(1, 2, MessageClass.CTRL, b"x")
        packed = bytearray(f.pack())
        packed[3] = 99  # invalid MessageClass value
        # Also need to fix CRC or we'd get CRC error first — reseed it.
        new_crc = crc16_ccitt(bytes(packed[: HEADER_SIZE + len(f.payload)]))
        packed[-CRC_SIZE:] = struct.pack(">H", new_crc)
        with pytest.raises(FrameError, match="unknown message_class"):
            Frame.unpack(bytes(packed))


# ---------------------------------------------------------------------------
# msgpack helpers
# ---------------------------------------------------------------------------
class TestMsgpackHelpers:
    @pytest.mark.parametrize(
        "obj",
        [
            None,
            True,
            False,
            0,
            -1,
            42,
            3.14,
            "hello",
            b"raw bytes",
            [],
            [1, 2, 3],
            {"key": "value"},
            {"nested": {"a": [1, 2], "b": None}},
        ],
    )
    def test_round_trip_common_types(self, obj: object) -> None:
        assert decode_payload(encode_payload(obj)) == obj

    def test_int_keys_supported(self) -> None:
        # strict_map_key=False lets us use int keys — useful for
        # compact enum-tagged payloads.
        obj = {1: "sos", 2: "ctrl"}
        assert decode_payload(encode_payload(obj)) == obj

    def test_control_frame_payload_is_compact(self) -> None:
        # A realistic RoboMaC destination command: goal coords + action.
        payload = {
            "type": "goto",
            "x": 42.1234567,
            "y": -83.9876543,
            "vmax": 0.5,
        }
        encoded = encode_payload(payload)
        # Should fit within a mesh payload with lots of headroom.
        assert len(encoded) < 60
        assert len(encoded) < MAX_PAYLOAD_BYTES


# ---------------------------------------------------------------------------
# End-to-end: msgpack payload inside a Frame
# ---------------------------------------------------------------------------
class TestFrameWithMsgpackPayload:
    def test_control_frame_round_trip(self) -> None:
        payload_dict = {"type": "goto", "x": 42.123, "y": -83.987}
        f = Frame.new(
            src_id=1,
            dst_id=42,
            message_class=MessageClass.CTRL,
            payload=encode_payload(payload_dict),
            seq=7,
            timestamp_ms=1_700_000_000_000,
        )
        packed = f.pack()
        recovered = Frame.unpack(packed)
        assert decode_payload(recovered.payload) == payload_dict
        assert recovered.seq == 7
        assert recovered.timestamp_ms == 1_700_000_000_000

    def test_sos_frame_from_emerg_device(self) -> None:
        # Mimics EMERG's SOS payload from Algorithm 1: victim lat/lon.
        payload_dict = {"lat": 42.4130, "lon": -83.1360, "id": "victim-01"}
        f = Frame.new(
            src_id=0x1001,
            dst_id=BROADCAST_ID,  # anyone who can hear, relay it
            message_class=MessageClass.SOS,
            payload=encode_payload(payload_dict),
            requires_ack=False,
            ttl=8,
        )
        recovered = Frame.unpack(f.pack())
        assert recovered.message_class == MessageClass.SOS
        assert recovered.is_broadcast()
        assert decode_payload(recovered.payload) == payload_dict


# ---------------------------------------------------------------------------
# Relay pattern: dataclasses.replace on frozen frame
# ---------------------------------------------------------------------------
class TestRelayPattern:
    def test_replace_produces_new_frame_with_decremented_ttl(self) -> None:
        f = Frame.new(1, 2, MessageClass.CTRL, b"x", ttl=8, hop_count=0)
        relayed = dataclasses.replace(
            f, ttl=f.ttl - 1, hop_count=f.hop_count + 1, prev_hop_id=99
        )
        assert relayed.ttl == 7
        assert relayed.hop_count == 1
        assert relayed.prev_hop_id == 99
        # Original untouched.
        assert f.ttl == 8
        assert f.hop_count == 0

    def test_replace_produces_valid_packable_frame(self) -> None:
        f = Frame.new(1, 2, MessageClass.CTRL, b"payload", ttl=8)
        relayed = dataclasses.replace(f, ttl=f.ttl - 1)
        # Should round-trip cleanly.
        assert Frame.unpack(relayed.pack()) == relayed
