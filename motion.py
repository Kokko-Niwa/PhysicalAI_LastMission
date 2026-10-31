"""
motion.py  —  ブルドーザーの「移動のしかた」だけをまとめた部品

使う側(sorter.py)が self.x, self.y, self.yaw（現在の位置と向き）と
self.spin()（最新の位置を受け取る）, self.send(v, w)（速度を送る）を用意すれば動く。
"""
import math
import time


def wrap(a):
    """角度を -π〜π の範囲にそろえる"""
    return math.atan2(math.sin(a), math.cos(a))


class MotionMixin:
    MAX_W = 1.0     # 最大の回転速度 [rad/s]
    MIN_W = 0.15    # これより遅いと回り切れないので下限を付ける

    def stop(self):
        for _ in range(3):
            self.send(0.0, 0.0)
            self.spin()

    def rotate_to(self, yaw_target, tol=0.03, timeout=20.0):
        """その場で向きを変える"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.spin()
            err = wrap(yaw_target - self.yaw)
            if abs(err) < tol:
                break
            w = max(-self.MAX_W, min(self.MAX_W, 2.0 * err))
            if abs(w) < self.MIN_W:
                w = math.copysign(self.MIN_W, w)
            self.send(0.0, w)
        self.stop()

    def drive_to(self, tx, ty, speed, reverse=False, tol=0.03, timeout=60.0):
        """(tx, ty) まで走る。reverse=True ならバックで向かう"""
        hdg = math.atan2(ty - self.y, tx - self.x)
        if reverse:
            hdg = wrap(hdg + math.pi)
        self.rotate_to(hdg)

        t0 = time.time()
        while time.time() - t0 < timeout:
            self.spin()
            dx, dy = tx - self.x, ty - self.y
            dist = math.hypot(dx, dy)
            if dist < tol:
                break
            # 進行方向に対して目標がまだ前にあるか（行き過ぎ検出）
            along = dx * math.cos(self.yaw) + dy * math.sin(self.yaw)
            if reverse:
                along = -along
            if along < 0:
                break
            hdg = math.atan2(dy, dx)
            if reverse:
                hdg = wrap(hdg + math.pi)
            err = wrap(hdg - self.yaw)
            v = min(speed, 0.8 * dist + 0.05)
            w = max(-self.MAX_W, min(self.MAX_W, 2.5 * err)) if dist > 0.08 else 0.0
            self.send(-v if reverse else v, w)
        self.stop()
