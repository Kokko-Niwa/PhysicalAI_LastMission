#!/usr/bin/env python3
"""
sorter.py  —  PAI最終課題：AI仕分けブルドーザー（本体）

流れ（ブロック1個ごとに繰り返す）:
  1. 判定台(0.7, 0)に新しいブロックが現れる
  2. 前カメラの画像を CLIP（学習なし・推論だけ）に見せ、RULES の説明文と照らし合わせる
  3. 一番近い説明文に対応するゾーンを選ぶ
  4. ブロックの斜め後ろに回り込み、ゾーンへ向かってまっすぐ押す
  5. バックで戻り、ホーム(0, 0)で次のブロックを待つ

使い方（pai_venv に入った状態で、Gazebo と run_bridge.sh を動かしておく）:
    python3 ~/pai_final/sorter.py
"""
import csv
import math
import os
import subprocess
import sys
import time

import numpy as np
import open_clip
import rclpy
import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion import MotionMixin  # noqa: E402

# ===================== ここを書き換えると仕分けの基準が変わる =====================
# (CLIPに見せる説明文[英語], 送り先ゾーン)
RULES = [
    ("a photo of a red box", "red_zone"),
    ("a photo of a blue box", "blue_zone"),
]
# 流れてくるブロックの順番（AIには教えない。答え合わせ用）
SEQUENCE = ["red", "blue", "blue", "red", "blue", "red"]
# =================================================================================

WORLD = "sorting_arena"
ZONES = {"red_zone": (1.8, 0.9), "blue_zone": (1.8, -0.9)}
ANSWER = {"red": "red_zone", "blue": "blue_zone"}
COLORS = {"red": "0.85 0.1 0.1 1", "blue": "0.1 0.2 0.85 1"}
BLOCK_SPOT = (0.7, 0.0)   # ブロックが現れる判定台
HOME = (0.0, 0.0)         # ブルドーザーの待機位置
CONTACT = 0.32            # ブロック中心とロボ中心の距離（押している時）
STAGE = 0.62              # 押し始める前に構える距離（ガードの先端がブロックに当たらない距離）

MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"

LOG_DIR = os.path.expanduser("~/pai_final/log")


class Sorter(MotionMixin, Node):
    def __init__(self):
        Node.__init__(self, "sorter")
        self.x = self.y = self.yaw = 0.0
        self.have_odom = False
        self.latest_image = None

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, "/sorter/status", 10)
        self.create_subscription(Odometry, "/bulldozer/odom", self.on_odom, 10)
        self.create_subscription(Image, "/bulldozer/camera", self.on_image, 1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.say(f"CLIPを読み込み中…（{self.device}）")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        self.model = self.model.to(self.device).eval()
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        with torch.no_grad():
            t = self.model.encode_text(tokenizer([r[0] for r in RULES]).to(self.device))
            self.text_feat = t / t.norm(dim=-1, keepdim=True)

    # ---------- ROSとのやりとり ----------
    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.x, self.y = p.x, p.y
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.have_odom = True

    def on_image(self, msg):
        self.latest_image = msg

    def spin(self):
        rclpy.spin_once(self, timeout_sec=0.02)

    def send(self, v, w):
        m = Twist()
        m.linear.x = float(v)
        m.angular.z = float(w)
        self.cmd_pub.publish(m)

    def wait(self, sec):
        t0 = time.time()
        while time.time() - t0 < sec:
            self.spin()

    def say(self, text, overlay=None):
        """ターミナルに日本語、動画の字幕用に英語(overlay)を流す"""
        self.get_logger().info(text)
        if overlay is not None:
            self.status_pub.publish(String(data=overlay))

    # ---------- Gazebo へのブロック出し入れ ----------
    def gz_service(self, service, reqtype, req):
        cmd = ["gz", "service", "-s", service, "--reqtype", reqtype,
               "--reptype", "gz.msgs.Boolean", "--timeout", "3000", "--req", req]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return "true" in r.stdout

    def remove_model(self, name):
        return self.gz_service(f"/world/{WORLD}/remove", "gz.msgs.Entity",
                               f'name: "{name}" type: MODEL')

    def spawn_block(self, name, color):
        c = COLORS[color]
        sdf = (
            "<sdf version='1.9'><model name='" + name + "'><link name='link'>"
            "<inertial><mass>0.3</mass><inertia><ixx>0.00072</ixx><iyy>0.00072</iyy>"
            "<izz>0.00072</izz></inertia></inertial>"
            "<collision name='c'><geometry><box><size>0.12 0.12 0.12</size></box></geometry>"
            "<surface><friction><ode><mu>0.5</mu><mu2>0.5</mu2></ode></friction></surface>"
            "</collision>"
            "<visual name='v'><geometry><box><size>0.12 0.12 0.12</size></box></geometry>"
            "<material><ambient>" + c + "</ambient><diffuse>" + c + "</diffuse></material>"
            "</visual></link></model></sdf>"
        )
        req = (f'sdf: "{sdf}" name: "{name}" '
               f'pose: {{position: {{x: {BLOCK_SPOT[0]}, y: {BLOCK_SPOT[1]}, z: 0.065}}}}')
        return self.gz_service(f"/world/{WORLD}/create", "gz.msgs.EntityFactory", req)

    # ---------- AI（CLIP）で判定 ----------
    def image_to_pil(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        img = img[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        return PILImage.fromarray(img.copy())

    @torch.no_grad()
    def classify(self, pil):
        x = self.preprocess(pil).unsqueeze(0).to(self.device)
        f = self.model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        probs = (100.0 * f @ self.text_feat.T).softmax(dim=-1)[0].tolist()
        best = max(range(len(RULES)), key=lambda i: probs[i])
        return best, probs

    # ---------- 押し方の計算 ----------
    @staticmethod
    def plan(zone_xy):
        bx, by = BLOCK_SPOT
        zx, zy = zone_xy
        d = math.hypot(zx - bx, zy - by)
        ux, uy = (zx - bx) / d, (zy - by) / d
        stage = (bx - ux * STAGE, by - uy * STAGE)
        push_end = (zx - ux * CONTACT, zy - uy * CONTACT)
        return stage, push_end

    # ---------- 本番 ----------
    def run(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.say("位置情報を待っています…")
        while rclpy.ok() and not self.have_odom:
            self.spin()
        self.remove_model("block_test")  # 最初から置いてあるテスト用ブロックは片付ける
        self.drive_to(*HOME, speed=0.3) if math.hypot(self.x, self.y) > 0.05 else None
        self.rotate_to(0.0)
        self.say("準備完了！ 仕分けを始めます", "Ready. Start sorting!")
        self.wait(2.0)

        correct = 0
        with open(os.path.join(LOG_DIR, "results.csv"), "w", newline="") as fp:
            log = csv.writer(fp)
            log.writerow(["block", "truth", "chosen_rule", "zone", "prob", "correct"])

            for i, color in enumerate(SEQUENCE, start=1):
                name = f"block_{i}"
                self.say(f"[{i}/{len(SEQUENCE)}] 新しいブロックが来ました",
                         f"Block {i}/{len(SEQUENCE)}: new block arrived")
                if not self.spawn_block(name, color):
                    self.say("  ブロックを出せませんでした（gz service の失敗）")
                self.wait(2.0)  # 落ち着くのと、新しい画像が届くのを待つ

                if self.latest_image is None:
                    self.say("  カメラ画像が来ていません。run_bridge.sh は動いていますか？")
                    return
                pil = self.image_to_pil(self.latest_image)
                pil.save(os.path.join(LOG_DIR, f"{name}.png"))
                best, probs = self.classify(pil)
                text, zone = RULES[best]
                ok = zone == ANSWER[color]
                correct += ok
                detail = " / ".join(f"{r[0]} {p*100:.1f}%" for r, p in zip(RULES, probs))
                self.say(f"  CLIP: {detail}")
                self.say(f"  判定 → {zone} へ運びます" + ("" if ok else "（※答えと違う！）"),
                         f"CLIP: '{text}' ({probs[best]*100:.0f}%) -> {zone}")
                log.writerow([name, color, text, zone, f"{probs[best]:.3f}", int(ok)])

                stage, push_end = self.plan(ZONES[zone])
                self.drive_to(*stage, speed=0.25)                  # 斜め後ろに構える
                self.say("  押します", f"Pushing block {i} to {zone}")
                self.drive_to(*push_end, speed=0.2)                # ゾーンへ押す
                self.drive_to(*stage, speed=0.3, reverse=True)     # バックで戻る
                self.drive_to(*HOME, speed=0.3)                    # ホームへ
                self.rotate_to(0.0)

        summary = f"Done! {correct}/{len(SEQUENCE)} blocks sorted correctly"
        self.say(f"完了！ {correct}/{len(SEQUENCE)} 個を正しく仕分けました", summary)
        self.wait(3.0)


def main():
    rclpy.init()
    node = Sorter()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.send(0.0, 0.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
