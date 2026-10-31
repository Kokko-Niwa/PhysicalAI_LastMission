#!/usr/bin/env python3
"""
snapshot.py  —  ブルドーザーの前カメラと俯瞰カメラの映像を1枚ずつ保存する確認用スクリプト

使い方（別ターミナルで ros_gz_bridge を動かした状態で）:
    python3 ~/pai_final/snapshot.py
保存先:
    ~/pai_final/snap_front.png     （ブルドーザーの目線）
    ~/pai_final/snap_overhead.png  （撮影用の俯瞰カメラ）
"""
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

OUT_DIR = os.path.expanduser("~/pai_final")
TARGETS = {
    "/bulldozer/camera": "snap_front.png",
    "/overhead/image": "snap_overhead.png",
}


def to_bgr(msg: Image) -> np.ndarray:
    """ROSの画像メッセージを OpenCV で保存できる形(BGR)に直す"""
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = img[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding.lower() == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


class Snapshot(Node):
    def __init__(self):
        super().__init__("snapshot")
        self.saved = set()
        for topic, fname in TARGETS.items():
            self.create_subscription(
                Image, topic, lambda msg, t=topic, f=fname: self.on_image(msg, t, f), 1
            )

    def on_image(self, msg, topic, fname):
        if topic in self.saved:
            return
        path = os.path.join(OUT_DIR, fname)
        cv2.imwrite(path, to_bgr(msg))
        self.saved.add(topic)
        self.get_logger().info(f"保存しました: {path}  ({msg.width}x{msg.height}, {msg.encoding})")


def main():
    rclpy.init()
    node = Snapshot()
    node.get_logger().info("画像を待っています…（10秒たっても何も出なければ Ctrl+C）")
    while rclpy.ok() and len(node.saved) < len(TARGETS):
        rclpy.spin_once(node, timeout_sec=0.5)
    node.get_logger().info("2枚とも保存できました！")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
