"""Priority queue for outgoing mesh frames.

Semantics:

    - 5 message classes (see MessageClass). Lower value = higher priority.
    - Strict priority: while any SOS frame is pending, no CTRL frame
      dequeues; while any SOS or CTRL is pending, no TELEM; and so on.
    - Same-class ordering is FIFO.
    - No in-flight preemption. LoRa PHY cannot stop mid-transmission.
      Callers set is_busy on the transport; a higher-priority arrival
      simply waits for the current TX to complete.
    - Head-of-line blocking is intentional. If the head-of-queue frame
      is duty-cycle-blocked, we wait for it — we do NOT skip past to a
      lower-priority frame that would fit. This preserves priority
      correctness at the cost of some throughput; the trade is
      appropriate for a SAR-critical system where SOS-first matters
      more than utilisation.
    - The queue registers airtime against the duty bucket at the moment
      of dequeue, on behalf of the caller. Caller is responsible for the
      actual transmission (i.e. calling transport.send(frame.pack())).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import simpy

from titan_communication.mesh.frame import Frame, MessageClass
from titan_communication.radio.duty import DutyBucket

if TYPE_CHECKING:
    from simpy.events import Event

__all__ = [
    "PendingFrame",
    "PriorityQueue",
]

# The five classes in priority order. Materialised once, indexed by
# int value of MessageClass.
_CLASS_ORDER: tuple[MessageClass, ...] = (
    MessageClass.SOS,
    MessageClass.CTRL,
    MessageClass.TELEM,
    MessageClass.AUDIO,
    MessageClass.BULK,
)


@dataclass(frozen=True)
class PendingFrame:
    """A frame plus the airtime its transmission will consume.

    Airtime is passed in (not computed here) so the queue doesn't need
    to know which LoRaParams the transport uses. Compute it once with
    ``time_on_air_s(params, len(frame.pack()))`` when you push.
    """

    frame: Frame
    airtime_s: float
    enqueue_time_s: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.airtime_s <= 0:
            raise ValueError(f"airtime_s must be positive, got {self.airtime_s}")


class PriorityQueue:
    """Strict-priority, duty-cycle-aware queue for one node's outgoing frames.

    Parameters
    ----------
    env : simpy.Environment
        The simulation environment. Used only for SimPy events; the
        queue does not sample env.now directly (see clock).
    duty : DutyBucket
        The node's duty-cycle bucket. Consulted on every dequeue.
    clock : Callable[[], float]
        Returns the current time in seconds. Defaults to ``lambda: env.now``.
        Injection point so tests can drive time deterministically.
    """

    def __init__(
        self,
        env: simpy.Environment,
        duty: DutyBucket,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._env = env
        self._duty = duty
        self._clock: Callable[[], float] = clock if clock is not None else (
            lambda: float(env.now)
        )
        # One FIFO Store per class. Store gives us FIFO for free.
        self._stores: dict[MessageClass, simpy.Store] = {
            cls: simpy.Store(env) for cls in _CLASS_ORDER
        }
        # Fired whenever a frame arrives; get()'s retry loop waits on this.
        self._new_item: simpy.Event = env.event()
        # Fired whenever duty state may have changed enough to unblock
        # a waiter. Currently only signalled via wake_duty_waiters().
        self._duty_ready: simpy.Event = env.event()

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Total pending frames across all classes."""
        return sum(len(s.items) for s in self._stores.values())

    def size_by_class(self, cls: MessageClass) -> int:
        """Pending frames for a specific class."""
        return len(self._stores[cls].items)

    def peek(self) -> PendingFrame | None:
        """Return (without removing) the highest-priority pending frame, or None."""
        for cls in _CLASS_ORDER:
            items = self._stores[cls].items
            if items:
                pending: PendingFrame = items[0]
                return pending
        return None

    def is_empty(self) -> bool:
        return self.size() == 0

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def put(self, pending: PendingFrame) -> None:
        """Enqueue a frame. Non-blocking, always succeeds."""
        self._stores[pending.frame.message_class].put(pending)
        self._signal_new_item()

    def put_frame(self, frame: Frame, airtime_s: float) -> None:
        """Convenience: enqueue by (frame, airtime) directly."""
        self.put(
            PendingFrame(
                frame=frame,
                airtime_s=airtime_s,
                enqueue_time_s=self._clock(),
            )
        )

    # ------------------------------------------------------------------
    # Dequeue (SimPy process — returns an Event)
    # ------------------------------------------------------------------

    def get(self) -> simpy.events.Process:
        """Dequeue the next frame that is (a) top-priority and (b) duty-allowed.

        Returns a SimPy Process. ``yield queue.get()`` in a SimPy process
        resolves to the ``PendingFrame`` that was dequeued.

        Behaviour:
            - If the queue is empty, waits for the next put().
            - If the head-of-queue frame is duty-blocked, waits — does NOT
              skip past it. Waits until either enough time passes that
              duty allows the head frame, or a higher-priority frame
              arrives.
            - On successful dequeue, registers the frame's airtime
              against the duty bucket automatically.
        """
        return self._env.process(self._get_process())

    def _get_process(self) -> Generator[Event, object, PendingFrame]:
        while True:
            head = self.peek()
            if head is None:
                # Queue empty — wait for arrival.
                yield self._new_item
                self._reset_new_item()
                continue

            now = self._clock()
            if self._duty.can_transmit(now, head.airtime_s):
                # Dequeue for real and register airtime.
                dequeued = cast(PendingFrame, (yield self._stores[
                    head.frame.message_class
                ].get()))
                self._duty.register(now, dequeued.airtime_s)
                return dequeued

            # Head is duty-blocked. Wait for either enough elapsed time
            # or a higher-priority arrival (which becomes the new head).
            wait_s = self._duty.time_until_next_available(now, head.airtime_s)
            if wait_s is None:
                # Head frame can never be transmitted (dwell violation or
                # airtime > full window). Drop it so the queue doesn't
                # deadlock; caller can observe via the dropped return.
                # We intentionally raise here rather than silently
                # drop — SAR-critical, we want the caller to notice.
                raise _UndeliverableFrameError(head)

            timeout = self._env.timeout(wait_s)
            yield timeout | self._new_item
            # Loop back and re-check. Either the timeout fired (duty may
            # now allow), or a new item arrived (possibly higher-priority).
            self._reset_new_item()

    # ------------------------------------------------------------------
    # Internal event plumbing
    # ------------------------------------------------------------------

    def _signal_new_item(self) -> None:
        """Fire the new-item event so any waiting get() retries."""
        if not self._new_item.triggered:
            self._new_item.succeed()

    def _reset_new_item(self) -> None:
        """Prepare a fresh new-item event after one has been consumed."""
        if self._new_item.triggered:
            self._new_item = self._env.event()

    def wake_duty_waiters(self) -> None:
        """External hook: signal that duty state changed (e.g. window reset).

        Currently the queue re-derives duty state via the injected clock
        + duty bucket on each retry, so this is a no-op placeholder for
        future explicit signalling. Left in the interface so mesh stack
        code has a hook if needed later.
        """
        # Intentional no-op. Kept in interface for forward compatibility.


class _UndeliverableFrameError(RuntimeError):
    """Internal: head frame cannot ever be transmitted under current policy."""

    def __init__(self, pending: PendingFrame | str) -> None:
        if isinstance(pending, str):
            # SimPy re-raises via type(exc)(*exc.args) — args[0] is
            # the formatted string from the original raise.
            super().__init__(pending)
            self.pending: PendingFrame | None = None
        else:
            super().__init__(
                f"frame is undeliverable under current duty policy: {pending.frame!r}, "
                f"airtime={pending.airtime_s * 1000:.2f} ms"
            )
            self.pending = pending


# Typing shim for the generator return type on older mypy.
# Placed at the bottom so the `if TYPE_CHECKING:` at top stays terse.
from collections.abc import Generator  # noqa: E402
