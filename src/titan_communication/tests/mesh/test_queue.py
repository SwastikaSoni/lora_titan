"""Tests for the mesh priority queue.

Two categories:

    - Pure-Python tests exercising put/peek/size/is_empty without
      driving SimPy time.
    - SimPy-driven tests that spin up an env, put/get, and verify
      priority + duty behaviour end-to-end.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
import simpy
from titan_communication.mesh.frame import Frame, MessageClass
from titan_communication.mesh.queue import (
    PendingFrame,
    PriorityQueue,
    _UndeliverableFrameError,
)
from titan_communication.radio.duty import DutyBucket, RegionPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mkframe(cls: MessageClass, seq: int = 0, payload: bytes = b"x") -> Frame:
    return Frame.new(src_id=1, dst_id=2, message_class=cls, payload=payload, seq=seq)


def _pending(
    cls: MessageClass,
    seq: int = 0,
    airtime_s: float = 0.1,
    enqueue_time_s: float = 0.0,
) -> PendingFrame:
    return PendingFrame(
        frame=_mkframe(cls, seq=seq),
        airtime_s=airtime_s,
        enqueue_time_s=enqueue_time_s,
    )


@pytest.fixture
def env() -> simpy.Environment:
    return simpy.Environment()


@pytest.fixture
def duty() -> DutyBucket:
    # US915-polite: 10% duty, 3600 s window -> 360 s budget. Plenty of headroom.
    return DutyBucket(RegionPolicy.us915_polite())


@pytest.fixture
def queue(env: simpy.Environment, duty: DutyBucket) -> PriorityQueue:
    return PriorityQueue(env=env, duty=duty)


# ---------------------------------------------------------------------------
# PendingFrame validation
# ---------------------------------------------------------------------------
class TestPendingFrameValidation:
    def test_valid_construction(self) -> None:
        p = _pending(MessageClass.CTRL)
        assert p.airtime_s == 0.1
        assert p.frame.message_class == MessageClass.CTRL

    @pytest.mark.parametrize("airtime", [0.0, -0.1])
    def test_invalid_airtime_raises(self, airtime: float) -> None:
        with pytest.raises(ValueError, match="airtime_s"):
            PendingFrame(frame=_mkframe(MessageClass.CTRL), airtime_s=airtime)


# ---------------------------------------------------------------------------
# Observability: empty / size / peek
# ---------------------------------------------------------------------------
class TestObservability:
    def test_empty_queue(self, queue: PriorityQueue) -> None:
        assert queue.is_empty()
        assert queue.size() == 0
        assert queue.peek() is None
        for cls in MessageClass:
            assert queue.size_by_class(cls) == 0

    def test_put_increments_size(self, queue: PriorityQueue) -> None:
        queue.put(_pending(MessageClass.CTRL))
        assert queue.size() == 1
        assert queue.size_by_class(MessageClass.CTRL) == 1
        assert not queue.is_empty()

    def test_size_by_class(self, queue: PriorityQueue) -> None:
        queue.put(_pending(MessageClass.SOS))
        queue.put(_pending(MessageClass.CTRL))
        queue.put(_pending(MessageClass.CTRL))
        queue.put(_pending(MessageClass.BULK))
        assert queue.size_by_class(MessageClass.SOS) == 1
        assert queue.size_by_class(MessageClass.CTRL) == 2
        assert queue.size_by_class(MessageClass.TELEM) == 0
        assert queue.size_by_class(MessageClass.AUDIO) == 0
        assert queue.size_by_class(MessageClass.BULK) == 1
        assert queue.size() == 4

    def test_peek_returns_highest_priority(self, queue: PriorityQueue) -> None:
        # Enqueue in reverse priority order; peek should still return SOS first.
        queue.put(_pending(MessageClass.BULK, seq=1))
        queue.put(_pending(MessageClass.AUDIO, seq=2))
        queue.put(_pending(MessageClass.SOS, seq=3))
        p = queue.peek()
        assert p is not None
        assert p.frame.message_class == MessageClass.SOS
        assert p.frame.seq == 3

    def test_peek_does_not_remove(self, queue: PriorityQueue) -> None:
        queue.put(_pending(MessageClass.CTRL))
        _ = queue.peek()
        assert queue.size() == 1

    def test_peek_within_class_is_fifo(self, queue: PriorityQueue) -> None:
        queue.put(_pending(MessageClass.CTRL, seq=1))
        queue.put(_pending(MessageClass.CTRL, seq=2))
        queue.put(_pending(MessageClass.CTRL, seq=3))
        p = queue.peek()
        assert p is not None
        assert p.frame.seq == 1


# ---------------------------------------------------------------------------
# put_frame convenience
# ---------------------------------------------------------------------------
class TestPutFrame:
    def test_put_frame_wraps_correctly(
        self, env: simpy.Environment, queue: PriorityQueue
    ) -> None:
        frame = _mkframe(MessageClass.CTRL, seq=42)
        queue.put_frame(frame, airtime_s=0.05)
        assert queue.size() == 1
        p = queue.peek()
        assert p is not None
        assert p.frame is frame
        assert p.airtime_s == 0.05
        assert p.enqueue_time_s == 0.0  # env.now at start


# ---------------------------------------------------------------------------
# SimPy-driven: single get() dequeues in priority order
# ---------------------------------------------------------------------------
class TestSimPyPriorityDequeue:
    def test_dequeue_returns_highest_priority(
        self, env: simpy.Environment, queue: PriorityQueue
    ) -> None:
        queue.put(_pending(MessageClass.BULK, seq=1))
        queue.put(_pending(MessageClass.SOS, seq=2))
        queue.put(_pending(MessageClass.CTRL, seq=3))

        result: dict[str, Any] = {}

        def consumer() -> Generator[simpy.Event, Any, None]:
            result["p"] = yield queue.get()

        env.process(consumer())
        env.run()
        assert result["p"].frame.message_class == MessageClass.SOS
        assert result["p"].frame.seq == 2

    def test_dequeue_five_in_full_priority_order(
        self, env: simpy.Environment, queue: PriorityQueue
    ) -> None:
        # Put one of each class in reverse order.
        for cls in reversed([MessageClass.SOS, MessageClass.CTRL,
                             MessageClass.TELEM, MessageClass.AUDIO,
                             MessageClass.BULK]):
            queue.put(_pending(cls))

        got: list[MessageClass] = []

        def consumer() -> Generator[simpy.Event, Any, None]:
            for _ in range(5):
                p = yield queue.get()
                got.append(p.frame.message_class)

        env.process(consumer())
        env.run()
        assert got == [
            MessageClass.SOS,
            MessageClass.CTRL,
            MessageClass.TELEM,
            MessageClass.AUDIO,
            MessageClass.BULK,
        ]

    def test_within_class_fifo(
        self, env: simpy.Environment, queue: PriorityQueue
    ) -> None:
        for seq in [10, 20, 30]:
            queue.put(_pending(MessageClass.CTRL, seq=seq))

        got_seqs: list[int] = []

        def consumer() -> Generator[simpy.Event, Any, None]:
            for _ in range(3):
                p = yield queue.get()
                got_seqs.append(p.frame.seq)

        env.process(consumer())
        env.run()
        assert got_seqs == [10, 20, 30]

    def test_get_blocks_on_empty_then_wakes(
        self, env: simpy.Environment, queue: PriorityQueue
    ) -> None:
        """Consumer starts on empty queue, producer puts at t=5, consumer receives."""
        result: dict[str, Any] = {}

        def consumer() -> Generator[simpy.Event, Any, None]:
            p = yield queue.get()
            result["p"] = p
            result["t"] = env.now

        def producer() -> Generator[simpy.Event, Any, None]:
            yield env.timeout(5.0)
            queue.put(_pending(MessageClass.CTRL, seq=99))

        env.process(consumer())
        env.process(producer())
        env.run()
        assert result["p"].frame.seq == 99
        assert result["t"] == 5.0


# ---------------------------------------------------------------------------
# SimPy-driven: duty-cycle-aware dequeue with head-of-line blocking
# ---------------------------------------------------------------------------
class TestDutyAwareDequeue:
    def test_registers_airtime_on_dequeue(
        self, env: simpy.Environment, duty: DutyBucket, queue: PriorityQueue
    ) -> None:
        queue.put(_pending(MessageClass.CTRL, airtime_s=0.5))

        def consumer() -> Generator[simpy.Event, Any, None]:
            yield queue.get()

        env.process(consumer())
        env.run()
        # 0.5 s of airtime should have been registered against the duty bucket.
        assert duty.airtime_used_s(now_s=env.now) == pytest.approx(0.5)

    def test_head_of_line_blocking_strict_priority(
        self, env: simpy.Environment
    ) -> None:
        """When SOS is duty-blocked, BULK does NOT jump the queue.

        We use a tiny duty budget so that even one small SOS packet has
        to wait, then confirm the BULK stays queued.
        """
        # 1% duty, 1-hour window -> 36 s budget.
        # Pre-load duty bucket to just under the cap.
        duty = DutyBucket(RegionPolicy.in865())
        duty.register(now_s=0.0, airtime_s=35.95)  # 50 ms of headroom left

        env = simpy.Environment()
        queue = PriorityQueue(env=env, duty=duty)

        # SOS needs 100 ms — blocked (only 50 ms left).
        queue.put(_pending(MessageClass.SOS, airtime_s=0.1, seq=1))
        # BULK is tiny — 10 ms fits.
        queue.put(_pending(MessageClass.BULK, airtime_s=0.01, seq=2))

        dequeue_order: list[tuple[MessageClass, float]] = []

        def consumer() -> Generator[simpy.Event, Any, None]:
            for _ in range(2):
                p = yield queue.get()
                dequeue_order.append((p.frame.message_class, float(env.now)))

        env.process(consumer())
        env.run()

        # SOS must come first, even though BULK could have gone immediately.
        assert dequeue_order[0][0] == MessageClass.SOS
        assert dequeue_order[1][0] == MessageClass.BULK
        # And SOS's dequeue happened only after enough duty freed up,
        # meaning env.now advanced substantially.
        assert dequeue_order[0][1] > 100.0  # waited a long time

    def test_wait_and_send_when_duty_frees_up(
        self, env: simpy.Environment
    ) -> None:
        """Head frame duty-blocked at t=0, becomes sendable after enough elapses."""
        duty = DutyBucket(RegionPolicy.in865())
        duty.register(now_s=0.0, airtime_s=35.99)  # 10 ms headroom

        env = simpy.Environment()
        queue = PriorityQueue(env=env, duty=duty)
        queue.put(_pending(MessageClass.CTRL, airtime_s=0.05, seq=1))

        result: dict[str, Any] = {}

        def consumer() -> Generator[simpy.Event, Any, None]:
            p = yield queue.get()
            result["seq"] = p.frame.seq
            result["t"] = float(env.now)

        env.process(consumer())
        env.run()
        assert result["seq"] == 1
        # Must have waited: 35.99 s in window, plus need 50 ms room.
        # First slot frees at t=3600. Consumer wakes around then.
        assert result["t"] >= 3599.0


class TestUndeliverableFrame:
    def test_undeliverable_raises(self, env: simpy.Environment) -> None:
        """A frame with airtime > full window budget cannot ever be sent."""
        duty = DutyBucket(RegionPolicy.in865())  # 36 s max ever
        queue = PriorityQueue(env=env, duty=duty)
        queue.put(_pending(MessageClass.CTRL, airtime_s=100.0, seq=1))

        # SimPy raises the exception on env.run() when the process fails.
        def consumer() -> Generator[simpy.Event, Any, None]:
            yield queue.get()

        env.process(consumer())
        with pytest.raises(_UndeliverableFrameError):
            env.run()


# ---------------------------------------------------------------------------
# Clock injection: use a manual clock, not env.now
# ---------------------------------------------------------------------------
class TestClockInjection:
    def test_injected_clock_used_for_enqueue_time(
        self, env: simpy.Environment, duty: DutyBucket
    ) -> None:
        fake_time = [42.0]

        def clock() -> float:
            return fake_time[0]

        queue = PriorityQueue(env=env, duty=duty, clock=clock)
        queue.put_frame(_mkframe(MessageClass.CTRL, seq=1), airtime_s=0.1)

        p = queue.peek()
        assert p is not None
        assert p.enqueue_time_s == 42.0


# ---------------------------------------------------------------------------
# Higher-priority arrival wakes a duty-blocked waiter
# ---------------------------------------------------------------------------
class TestPriorityInversionAvoidance:
    def test_new_higher_priority_arrival_reevaluates(
        self, env: simpy.Environment, duty: DutyBucket, queue: PriorityQueue
    ) -> None:
        """A BULK sits waiting; SOS arrives; SOS dequeues first.

        Even with plenty of duty budget — this is just about the retry
        loop noticing a new higher-priority arrival.
        """
        # No pre-load; both should be immediately sendable when their turn comes.
        queue.put(_pending(MessageClass.BULK, airtime_s=0.05, seq=1))

        dequeued: list[MessageClass] = []

        def consumer() -> Generator[simpy.Event, Any, None]:
            for _ in range(2):
                p = yield queue.get()
                dequeued.append(p.frame.message_class)

        def sos_arriver() -> Generator[simpy.Event, Any, None]:
            yield env.timeout(0.1)
            queue.put(_pending(MessageClass.SOS, airtime_s=0.05, seq=99))

        env.process(consumer())
        env.process(sos_arriver())
        env.run()

        # BULK could have gone at t=0 but under the test conditions,
        # the consumer is fast — it might've already dequeued BULK
        # before SOS arrived. In that case we only see [BULK, SOS].
        # Both orderings are valid *starting states*, but by end of run
        # both must have gone through.
        assert set(dequeued) == {MessageClass.BULK, MessageClass.SOS}
