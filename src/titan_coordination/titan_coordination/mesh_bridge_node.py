"""
mesh_bridge_node.py — Weeks 7-8, Task 3: leader broadcasts LoRa telemetry.

Owns the one shared SimPy environment + VirtualChannel that every mesh
participant (leader, BS, and later followers) must share to exchange
frames — VirtualChannel.broadcast() schedules delivery via
env.process(), so every registered transport has to be stepped by the
same SimPy env. This node is that shared process, bridged into ROS by
advancing the env by the elapsed *sim* time on every timer tick
(use_sim_time=True), so LoRa airtime/duty-cycle accounting stays on the
same clock as Gazebo.

Data path:
    /leader/mesh_tx (msgpack bytes, published by platoon_leader.py)
        -> leader_stack.send(dst=BS, TELEM, payload)
        -> VirtualChannel computes RSSI from the leader's live position
           (subscribed from /leader/odometry/filtered) to the BS's fixed
           position
        -> delivered to the BS's raw transport receive callback,
           registered directly rather than through routing's dst-match
           (flood routing rewrites every frame's dst to broadcast, so
           routing-level DELIVER never fires for a specific address —
           see bakeoff.py's make_bs_rx_sniffer for the same pattern)
        -> published on /mesh/bs_rx as JSON for bs_sink_node.py to log.

Only two mesh addresses exist today: BS=0x0001, leader=0x0002.
0x0003-0x0006 are reserved for follower_1..4 (Task 5+).
"""

from __future__ import annotations

import json
import os

import numpy as np
import rclpy
import simpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray

from titan_communication.mesh.frame import BROADCAST_ID, Frame, MessageClass, decode_payload
from titan_communication.mesh.queue import PriorityQueue
from titan_communication.mesh.routing_flood import FloodRouting
from titan_communication.mesh.stack import MeshStack
from titan_communication.radio.airtime import LoRaParams
from titan_communication.radio.channel import ChannelModel
from titan_communication.radio.duty import DutyBucket, RegionPolicy
from titan_communication.radio.transport import ReceptionInfo
from titan_communication.radio.virtual import VirtualChannel, VirtualLoRaTransport

BS_ADDR = 0x0001
LEADER_ADDR = 0x0002

_DUTY_POLICIES = {
    "us915_polite": RegionPolicy.us915_polite,
    "us915_fcc_raw": RegionPolicy.us915_fcc_raw,
    "us915_hopping": RegionPolicy.us915_hopping,
    "in865": RegionPolicy.in865,
    "eu868": RegionPolicy.eu868,
}


class MeshBridgeNode(Node):
    """Shared SimPy mesh environment (leader + BS) bridged into ROS."""

    def __init__(self) -> None:
        super().__init__("mesh_bridge")

        self.declare_parameter("configs_dir", os.path.expanduser("~/titan_ws/configs"))
        self.declare_parameter("lora_preset", "paper_default")
        self.declare_parameter("channel_model", "outdoor_suburban")
        self.declare_parameter("duty_policy", "us915_polite")
        self.declare_parameter("tx_power_dbm", 15.0)
        self.declare_parameter("bs_x", 65.0)
        self.declare_parameter("bs_y", 0.0)
        self.declare_parameter("sim_step_frequency", 20.0)

        configs_dir = str(self.get_parameter("configs_dir").value)
        lora_preset = str(self.get_parameter("lora_preset").value)
        channel_model_name = str(self.get_parameter("channel_model").value)
        duty_policy_name = str(self.get_parameter("duty_policy").value)
        tx_power_dbm = float(self.get_parameter("tx_power_dbm").value)
        self._bs_pos = (
            float(self.get_parameter("bs_x").value),
            float(self.get_parameter("bs_y").value),
        )
        sim_step_frequency = float(self.get_parameter("sim_step_frequency").value)

        lora_params = LoRaParams.from_preset(
            os.path.join(configs_dir, "lora_params.yaml"), lora_preset
        )
        channel_model = ChannelModel.from_preset(
            os.path.join(configs_dir, "channel.yaml"), channel_model_name
        )
        duty_region = _DUTY_POLICIES[duty_policy_name]()
        rng = np.random.default_rng()

        # ── Shared SimPy world ──
        self._env = simpy.Environment()
        self._sim_channel = VirtualChannel(self._env, channel_model, rng)

        # ── Leader participant ──
        self._leader_pos = (0.0, 0.0)
        leader_transport = VirtualLoRaTransport(
            env=self._env,
            channel=self._sim_channel,
            node_id="leader",
            params=lora_params,
            duty=DutyBucket(duty_region),
            position_getter=lambda: self._leader_pos,
            tx_power_dbm=tx_power_dbm,
        )
        leader_routing = FloodRouting(node_addr=LEADER_ADDR, bs_addr=BS_ADDR, is_bs=False)
        leader_queue = PriorityQueue(self._env, DutyBucket(duty_region))
        self._leader_stack = MeshStack(
            env=self._env,
            transport=leader_transport,
            routing=leader_routing,
            queue=leader_queue,
            lora_params=lora_params,
            node_addr=LEADER_ADDR,
            bs_addr=BS_ADDR,
            is_bs=False,
        )

        # ── BS participant ──
        # BROADCAST_ID as bs_addr: flood routing overwrites every frame's
        # dst to broadcast before it ever goes on air (see module
        # docstring), so a normal address never matches at the BS's own
        # routing layer. This makes the BS's *own* MeshStack.on_deliver
        # fire for broadcast TELEM/SOS/etc. too, in addition to the raw
        # sniffer below (which is what Task 3 actually verifies against).
        bs_transport = VirtualLoRaTransport(
            env=self._env,
            channel=self._sim_channel,
            node_id="bs",
            params=lora_params,
            duty=DutyBucket(duty_region),
            position_getter=lambda: self._bs_pos,
            tx_power_dbm=tx_power_dbm,
        )
        bs_routing = FloodRouting(node_addr=BS_ADDR, bs_addr=BROADCAST_ID, is_bs=True)
        bs_queue = PriorityQueue(self._env, DutyBucket(duty_region))
        self._bs_stack = MeshStack(
            env=self._env,
            transport=bs_transport,
            routing=bs_routing,
            queue=bs_queue,
            lora_params=lora_params,
            node_addr=BS_ADDR,
            bs_addr=BROADCAST_ID,
            is_bs=True,
        )

        # Raw sniffer — fires on every packet physically reaching the BS,
        # regardless of the mesh layer's dst-match logic. Registered
        # alongside (not instead of) the stack's own on_receive.
        bs_transport.on_receive(self._on_bs_rx)

        self._leader_stack.start()
        self._bs_stack.start()

        # ── ROS I/O ──
        self.create_subscription(
            Odometry, "/leader/odometry/filtered", self._leader_odom_cb, 10
        )
        self.create_subscription(UInt8MultiArray, "/leader/mesh_tx", self._mesh_tx_cb, 10)
        self._bs_rx_pub = self.create_publisher(String, "/mesh/bs_rx", 10)

        self._last_tick = self.get_clock().now()
        self.create_timer(1.0 / sim_step_frequency, self._step_sim)

        self.get_logger().info(
            f"mesh_bridge up: leader=0x{LEADER_ADDR:04x} @ (0,0), "
            f"bs=0x{BS_ADDR:04x} @ {self._bs_pos}, "
            f"lora={lora_preset}, channel={channel_model_name}, duty={duty_policy_name}"
        )

    def _leader_odom_cb(self, msg: Odometry) -> None:
        self._leader_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _mesh_tx_cb(self, msg: UInt8MultiArray) -> None:
        self._leader_stack.send(
            dst_addr=BS_ADDR,
            message_class=MessageClass.TELEM,
            payload=bytes(msg.data),
        )

    def _on_bs_rx(self, data: bytes, info: ReceptionInfo) -> None:
        try:
            frame = Frame.unpack(data)
        except Exception as e:  # noqa: BLE001 — log and move on, not fatal
            self.get_logger().warning(f"BS rx: unpack failed: {e}")
            return

        try:
            payload = decode_payload(frame.payload)
        except Exception:
            payload = None

        out = String()
        out.data = json.dumps(
            {
                "src": frame.src_id,
                "message_class": frame.message_class.name,
                "rssi_dbm": info.rssi_dbm,
                "snr_db": info.snr_db,
                "payload": payload,
            }
        )
        self._bs_rx_pub.publish(out)

    def _step_sim(self) -> None:
        now = self.get_clock().now()
        dt = (now - self._last_tick).nanoseconds / 1e9
        self._last_tick = now
        if dt <= 0.0:
            return
        self._env.run(until=self._env.now + dt)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MeshBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
