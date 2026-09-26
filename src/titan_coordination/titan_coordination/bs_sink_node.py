"""
bs_sink_node.py — Weeks 7-8, Task 3 verification: does the BS receive it?

Subscribes to /mesh/bs_rx (published by mesh_bridge_node's raw BS
receive sniffer) and logs each arriving frame: source address, message
class, RSSI/SNR from the virtual channel, and the decoded payload.
Purely a verification/logging node — nothing downstream depends on it
yet (Task 7's health_score will read RSSI trend from here later).
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class BsSinkNode(Node):
    def __init__(self) -> None:
        super().__init__("bs_sink")
        self.create_subscription(String, "/mesh/bs_rx", self._on_rx, 10)
        self.get_logger().info("bs_sink up, listening on /mesh/bs_rx")

    def _on_rx(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warning(f"malformed /mesh/bs_rx payload: {e}")
            return

        self.get_logger().info(
            f"BS received {data['message_class']} from 0x{data['src']:04x}: "
            f"rssi={data['rssi_dbm']:.1f} dBm  snr={data['snr_db']:.1f} dB  "
            f"payload={data['payload']}"
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = BsSinkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
