
from __future__ import annotations

from dataclasses import replace
from enum import Enum, auto
from typing import Any, Callable, Optional, List

import numpy as np
import simpy

from .frame import (
    Frame,
    MessageClass,
    BROADCAST_ID,
)
from .routing import (
    IRoutingScheme,
    RoutingAction,
    ActionType,
)

from ..radio.airtime import LoRaParams, time_on_air_s
from ..radio.transport import ReceptionInfo


# ---------------------------------------------------------------------------
# Event types for the per-packet log
# ---------------------------------------------------------------------------

class LogEvent(str, Enum):
    TX = "tx"
    RX = "rx"
    DELIVER = "deliver"
    FORWARD = "forward"
    DROP = "drop"
    ENQUEUE = "enqueue"
    DUTY_WAIT = "duty_wait"
    CTRL_TX = "ctrl_tx"
    BUSY_WAIT = "busy_wait"


# ---------------------------------------------------------------------------
# MeshStack
# ---------------------------------------------------------------------------

class MeshStack:

    def __init__(
        self,
        env: simpy.Environment,
        transport: Any,          # VirtualLoRaTransport instance
        routing: IRoutingScheme,
        queue: Any,              # PriorityQueue instance
        lora_params: LoRaParams,
        node_addr: int,
        bs_addr: int = 0x0001,
        is_bs: bool = False,
        tick_interval_s: float = 1.0,
        busy_poll_s: float = 0.01,
        on_deliver: Optional[Callable[[Frame, float], None]] = None,
        backoff_max_s: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Args:
            env:              SimPy Environment.
            transport:        VirtualLoRaTransport for this node.
            routing:          IRoutingScheme instance (flood/aodv/gradient).
            queue:            PriorityQueue instance. Should be constructed
                              with a permissive DutyBucket (no cap) so the
                              queue does pure priority ordering. Duty
                              enforcement is handled by the transport.
            lora_params:      LoRaParams for airtime calculation.
            node_addr:        This node's 16-bit mesh address.
            bs_addr:          Base station address.
            is_bs:            True if this node is the BS.
            tick_interval_s:  How often to call routing.on_tick().
            busy_poll_s:      How often to poll transport.is_busy.
            on_deliver:       Optional callback when a frame is delivered to app.
            backoff_max_s:    Random channel-access backoff applied before
                              every transmission, drawn uniformly from
                              [0, backoff_max_s). Without this, nodes that
                              react to the same received broadcast in the
                              same simulated instant (e.g. an AODV
                              destination replying with RREP while its
                              other neighbours simultaneously rebroadcast
                              the same RREQ) transmit at the identical
                              timestamp and collide deterministically,
                              every time, in any topology where they share
                              radio range. 0 disables backoff.
            rng:              Generator driving the backoff draw. Required
                              (deterministically) when backoff_max_s > 0;
                              falls back to an unseeded default otherwise.
        """
        self.env = env
        self.transport = transport
        self.routing = routing
        self.queue = queue
        self.lora_params = lora_params
        self.node_addr = node_addr
        self.bs_addr = bs_addr
        self.is_bs = is_bs
        self.tick_interval_s = tick_interval_s
        self.busy_poll_s = busy_poll_s
        self.on_deliver = on_deliver
        self.backoff_max_s = backoff_max_s
        self.rng = rng if rng is not None else np.random.default_rng()

        # Bridge: transport callback -> SimPy Store for the RX loop
        self._rx_store: simpy.Store = simpy.Store(env)

        # Per-packet event log — bakeoff reads this
        self.log: list[dict] = []

        # Counters
        self.tx_count: int = 0
        self.rx_count: int = 0
        self.deliver_count: int = 0
        self.drop_count: int = 0
        self.forward_count: int = 0
        self.busy_wait_count: int = 0

        # SimPy processes
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch SimPy processes and register transport callback."""
        self._running = True

        # Register the RX callback on the transport — bridges into SimPy
        self.transport.on_receive(self._on_rx_callback)

        self.env.process(self._rx_loop())
        self.env.process(self._tx_loop())
        self.env.process(self._tick_loop())

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Transport RX callback -> SimPy Store bridge
    # ------------------------------------------------------------------

    def _on_rx_callback(self, data: bytes, info: ReceptionInfo) -> None:
        """Called by VirtualLoRaTransport when a packet arrives.
        Puts into the SimPy Store so _rx_loop can yield on it."""
        self._rx_store.put((data, info))

    # ------------------------------------------------------------------
    # App-layer send API
    # ------------------------------------------------------------------

    def send(
        self,
        dst_addr: int,
        message_class: MessageClass,
        payload: bytes,
        requires_ack: bool = False,
    ) -> None:
        """
        App layer wants to send a frame. Asks routing for next-hop,
        builds the Frame, enqueues it.

        Call from a SimPy process or from the bakeoff traffic generator.
        """
        now = self.env.now

        action = self.routing.get_next_hop(
            dst_addr=dst_addr,
            tx_time=now,
        )

        if action.action == ActionType.DROP:
            self._log_event(LogEvent.DROP, now, {
                "reason": action.reason,
                "dst": dst_addr,
                "class": message_class.name,
            })
            self.drop_count += 1
            self._flush_control_frames(now)
            return

        # Build the Frame
        frame = Frame.new(
            src_id=self.node_addr,
            dst_id=dst_addr,
            message_class=message_class,
            payload=payload,
            seq=action.seq,
            ttl=action.ttl,
            prev_hop_id=self.node_addr,
            timestamp_ms=int(now * 1000),
            requires_ack=requires_ack,
        )

        if action.action == ActionType.DELIVER:
            self._deliver(frame, now)
            return

        # BROADCAST -> override dst_id
        if action.action == ActionType.BROADCAST:
            frame = replace(frame, dst_id=BROADCAST_ID)

        self._enqueue(frame, now)
        self._flush_control_frames(now)

    # ------------------------------------------------------------------
    # RX loop (SimPy process)
    # ------------------------------------------------------------------

    def _rx_loop(self):
        """Block on the rx_store, unpack, dispatch to routing."""
        while self._running:
            try:
                item = yield self._rx_store.get()
            except simpy.Interrupt:
                break

            data, info = item
            now = self.env.now
            self.rx_count += 1

            # Unpack the frame
            try:
                frame = Frame.unpack(data)
            except Exception as e:
                self._log_event(LogEvent.DROP, now, {
                    "reason": f"unpack_error: {e}",
                })
                self.drop_count += 1
                continue

            self._log_event(LogEvent.RX, now, {
                "src": frame.src_id,
                "dst": frame.dst_id,
                "prev_hop": frame.prev_hop_id,
                "seq": frame.seq,
                "class": frame.message_class.name,
                "rssi": info.rssi_dbm,
                "snr": info.snr_db,
                "ttl": frame.ttl,
                "hop_count": frame.hop_count,
            })

            # Hand to routing
            action = self.routing.on_frame_received(
                frame=frame,
                rssi_dbm=info.rssi_dbm,
                snr_db=info.snr_db,
                rx_time=now,
            )

            self._handle_routing_action(action, now)
            self._flush_control_frames(now)

    # ------------------------------------------------------------------
    # TX loop (SimPy process)
    # ------------------------------------------------------------------

    def _tx_loop(self):
        """Dequeue PendingFrames and transmit via transport."""
        while self._running:
            try:
                pending = yield self.queue.get()
            except simpy.Interrupt:
                break
            except Exception:
                # _UndeliverableFrameError — frame can never be sent
                # under current duty policy. Drop it.
                self.drop_count += 1
                continue

            frame = pending.frame

            # Random channel-access backoff — decorrelates nodes that react
            # to the same received broadcast in the same instant. Without
            # this, simultaneous reactive transmissions (e.g. an AODV RREP
            # and other neighbours' RREQ rebroadcasts, all triggered by the
            # same original frame) always collide.
            if self.backoff_max_s > 0:
                backoff = float(self.rng.uniform(0.0, self.backoff_max_s))
                try:
                    yield self.env.timeout(backoff)
                except simpy.Interrupt:
                    break

            packed = frame.pack()
            now = self.env.now

            # Wait for transport to not be busy (previous TX completing)
            while self.transport.is_busy:
                self.busy_wait_count += 1
                self._log_event(LogEvent.BUSY_WAIT, self.env.now, {
                    "seq": frame.seq,
                    "class": frame.message_class.name,
                })
                try:
                    yield self.env.timeout(self.busy_poll_s)
                except simpy.Interrupt:
                    return

            # Transmit — transport.send() is non-blocking
            try:
                self.transport.send(packed)
            except Exception as e:
                # DutyViolationError or TransportBusyError (race)
                self._log_event(LogEvent.DROP, self.env.now, {
                    "reason": f"tx_error: {e}",
                    "seq": frame.seq,
                    "class": frame.message_class.name,
                })
                self.drop_count += 1
                continue

            self.tx_count += 1
            self._log_event(LogEvent.TX, self.env.now, {
                "src": frame.src_id,
                "dst": frame.dst_id,
                "prev_hop": frame.prev_hop_id,
                "seq": frame.seq,
                "class": frame.message_class.name,
                "airtime_ms": pending.airtime_s * 1000,
                "size": len(packed),
                "ttl": frame.ttl,
                "hop_count": frame.hop_count,
            })

    # ------------------------------------------------------------------
    # Tick loop (SimPy process)
    # ------------------------------------------------------------------

    def _tick_loop(self):
        """Periodically call routing.on_tick() and flush control frames."""
        while self._running:
            try:
                yield self.env.timeout(self.tick_interval_s)
            except simpy.Interrupt:
                break

            now = self.env.now
            self.routing.on_tick(now)
            self._flush_control_frames(now)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_routing_action(self, action: RoutingAction, now: float) -> None:
        if action.action == ActionType.DELIVER:
            self._deliver(action.frame, now)

        elif action.action in (ActionType.FORWARD, ActionType.BROADCAST):
            if action.frame is not None:
                self._enqueue(action.frame, now)
                self.forward_count += 1
                self._log_event(LogEvent.FORWARD, now, {
                    "src": action.frame.src_id,
                    "dst": action.frame.dst_id,
                    "next_hop": action.next_hop,
                    "seq": action.frame.seq,
                    "reason": action.reason,
                })

        elif action.action == ActionType.DROP:
            self.drop_count += 1
            self._log_event(LogEvent.DROP, now, {
                "reason": action.reason,
            })

    def _deliver(self, frame: Optional[Frame], now: float) -> None:
        if frame is None:
            return
        self.deliver_count += 1
        self._log_event(LogEvent.DELIVER, now, {
            "src": frame.src_id,
            "dst": frame.dst_id,
            "seq": frame.seq,
            "class": frame.message_class.name,
            "hop_count": frame.hop_count,
            "latency_ms": now * 1000 - frame.timestamp_ms,
        })
        if self.on_deliver is not None:
            self.on_deliver(frame, now)

    def _enqueue(self, frame: Frame, now: float) -> None:
        """Compute airtime and enqueue as a PendingFrame via put_frame."""
        packed_len = frame.total_size()
        airtime = time_on_air_s(self.lora_params, packed_len)
        self.queue.put_frame(frame, airtime)
        self._log_event(LogEvent.ENQUEUE, now, {
            "seq": frame.seq,
            "class": frame.message_class.name,
            "dst": frame.dst_id,
            "airtime_ms": airtime * 1000,
        })

    def _flush_control_frames(self, now: float) -> None:
        ctrl_frames = self.routing.get_pending_control_frames()
        for cf in ctrl_frames:
            self._enqueue(cf, now)
            self._log_event(LogEvent.CTRL_TX, now, {
                "seq": cf.seq,
                "dst": cf.dst_id,
                "class": cf.message_class.name,
            })

    def _log_event(self, event: LogEvent, sim_time: float, data: dict) -> None:
        entry = {
            "time": sim_time,
            "node": self.node_addr,
            "event": event.value,
        }
        entry.update(data)
        self.log.append(entry)

    # ------------------------------------------------------------------
    # Metrics summary
    # ------------------------------------------------------------------

    def metrics_summary(self) -> dict:
        return {
            "node": self.node_addr,
            "is_bs": self.is_bs,
            "tx": self.tx_count,
            "rx": self.rx_count,
            "deliver": self.deliver_count,
            "drop": self.drop_count,
            "forward": self.forward_count,
            "busy_wait": self.busy_wait_count,
            "log_len": len(self.log),
            "neighbours": len(self.routing.neighbours),
        }