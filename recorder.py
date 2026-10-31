#!/usr/bin/env python3
"""
recorder.py  —  俯瞰カメラの映像を動画(mp4)に保存する。右上にロボ目線、下にAIの判定字幕を重ねる。

使い方（Gazebo と run_bridge.sh を動かしておく）:
    python3 ~/pai_final/recorder.py              # 録画＋プレビュー窓
    python3 ~/pai_final/recorder.py --no-preview # 窓が出せない時はこちら
止めるときは Ctrl+C（そこでファイルが閉じられる）
保存先: ~/pai_final/video/sorting_YYYYmmdd_HHMMSS.mp4
"""
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

FPS = 15  # 俯瞰カメラの update_rate と同じ
TITLE = "PAI Final: AI Sorting Bulldozer (CLIP zero-shot, no training)"
OUT_DIR = os.path.expanduser("~/pai_final/video")


def to_bgr(msg):
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = img[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding.lower() == "rgb8" else img.copy()


def put_label(img, text, org, scale=0.8, thick=2):
    """背景つきの読みやすい文字"""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 8, y - h - 10), (x + w + 8, y + base + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


class Recorder(Node):
    def __init__(self, preview):
        super().__init__("recorder")
        self.preview = preview
        self.front = None
        self.status = "Waiting..."
        self.writer = None
        os.makedirs(OUT_DIR, exist_ok=True)
        self.path = os.path.join(OUT_DIR, time.strftime("sorting_%Y%m%d_%H%M%S.mp4"))
        self.frames = 0
        self.create_subscription(Image, "/overhead/image", self.on_overhead, 5)
        self.create_subscription(Image, "/bulldozer/camera", self.on_front, 1)
        self.create_subscription(String, "/sorter/status", self.on_status, 10)

    def on_front(self, msg):
        self.front = to_bgr(msg)

    def on_status(self, msg):
        self.status = msg.data

    def on_overhead(self, msg):
        frame = to_bgr(msg)
        H, W = frame.shape[:2]
        if self.writer is None:
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
            self.get_logger().info(f"録画開始: {self.path}")

        # 右上にロボ目線（ワイプ）
        if self.front is not None:
            inset = cv2.resize(self.front, (384, 288))
            x0, y0 = W - 384 - 16, 16
            cv2.rectangle(frame, (x0 - 3, y0 - 3), (x0 + 384 + 3, y0 + 288 + 3), (255, 255, 255), -1)
            frame[y0:y0 + 288, x0:x0 + 384] = inset
            put_label(frame, "Robot camera -> CLIP", (x0 + 6, y0 + 288 + 30), 0.6, 1)

        put_label(frame, TITLE, (20, 40), 0.7, 2)
        put_label(frame, self.status, (20, H - 24), 0.9, 2)

        self.writer.write(frame)
        self.frames += 1
        if self.preview:
            try:
                cv2.imshow("recorder (Ctrl+C in terminal to stop)", cv2.resize(frame, (960, 540)))
                cv2.waitKey(1)
            except cv2.error:
                self.preview = False
                self.get_logger().warn("プレビュー窓が出せないので、録画だけ続けます")

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.get_logger().info(f"保存しました: {self.path}  ({self.frames} フレーム ≒ {self.frames / FPS:.0f} 秒)")
        if self.preview:
            cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = Recorder(preview="--no-preview" not in sys.argv)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
