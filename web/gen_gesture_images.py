# -*- coding: utf-8 -*-
"""
无声有译 · 词库配图生成器

从 sign_data/<词>/*.npy 里读样本, 画出手部骨架示意图, 存到 web/static/gestures/.
  <词>.png    -> 该词最好的一张(词库页封面)
  <词>_s1..3.png -> 前 3 条样本(词库页"查看样本"用)

样本格式: (帧数, 63) = 单手 21 点 × (x,y,z); 或 (帧数, 126) = 双手各 21 点。
坐标是"以手腕为原点、手腕到中指根为尺子"归一化后的值, 所以画出来是手的形状,
不带绝对位置——用来做配图正合适。

运行: D:/python/python.exe web/gen_gesture_images.py
依赖: numpy + Pillow (1.py 本来就在用, 不新增)
"""

import os
import glob
import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
SIGN_DATA = os.path.join(PROJ_DIR, "sign_data")
OUT_DIR = os.path.join(BASE_DIR, "static", "gestures")

CANVAS = 360
MARGIN = 46
BG = (13, 21, 32)
COL_MAIN = (34, 211, 238)     # 主手: 青色
COL_SECOND = (251, 191, 36)   # 另一只手: 琥珀色

# MediaPipe 21 点连线关系(手指骨架)
LINKS = [(0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16),
         (13, 17), (17, 18), (18, 19), (19, 20),
         (0, 17)]


def split_hands(frame):
    """一帧 -> [手1的21个(x,y), 手2的21个(x,y) 或 None]"""
    hands = []
    n = len(frame) // 63
    for i in range(min(n, 2)):
        blk = frame[i * 63:(i + 1) * 63].reshape(21, 3)
        if np.abs(blk).sum() < 1e-6:          # 这只手没检测到
            continue
        hands.append(blk[:, :2])
    return hands


def pick_frame(arr):
    """在样本里挑形状最舒展的一帧(比取中间帧稳, 避开过渡动作)"""
    best, best_area = 0, -1
    for i in range(len(arr)):
        hands = split_hands(arr[i])
        if not hands:
            continue
        pts = np.vstack(hands)
        w = pts[:, 0].max() - pts[:, 0].min()
        h = pts[:, 1].max() - pts[:, 1].min()
        area = w * h
        if area > best_area:
            best, best_area = i, area
    return best


def draw_frame(arr, idx, path):
    hands = split_hands(arr[idx])
    img = Image.new("RGB", (CANVAS, CANVAS), BG)
    if not hands:
        img.save(path)
        return False
    d = ImageDraw.Draw(img)

    allpts = np.vstack(hands)
    x0, y0 = allpts.min(axis=0)
    x1, y1 = allpts.max(axis=0)
    span = max(x1 - x0, y1 - y0, 1e-6)
    scale = (CANVAS - 2 * MARGIN) / span
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    def to_px(p):
        # y 轴: 样本里向上是负, 画布向下是正, 直接线性映射即正确方向
        return (CANVAS / 2 + (p[0] - cx) * scale,
                CANVAS / 2 + (p[1] - cy) * scale)

    for k, hand in enumerate(hands):
        col = COL_MAIN if k == 0 else COL_SECOND
        for a, b in LINKS:
            d.line([to_px(hand[a]), to_px(hand[b])], fill=col, width=7)
        for i, p in enumerate(hand):
            r = 9 if i == 0 else 5
            x, y = to_px(p)
            d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    img.save(path)
    return True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isdir(SIGN_DATA):
        print("找不到样本目录:", SIGN_DATA)
        return
    total = 0
    for word in sorted(os.listdir(SIGN_DATA)):
        d = os.path.join(SIGN_DATA, word)
        if not os.path.isdir(d):
            continue
        files = sorted(glob.glob(os.path.join(d, "*.npy")))
        if not files:
            print(f"  跳过 {word}: 没有样本")
            continue
        # 封面: 第一条样本
        arr = np.load(files[0]).astype(np.float32)
        ok = draw_frame(arr, pick_frame(arr), os.path.join(OUT_DIR, f"{word}.png"))
        total += 1 if ok else 0
        # 样本图: 前三条
        for k, f in enumerate(files[:3], 1):
            a = np.load(f).astype(np.float32)
            draw_frame(a, pick_frame(a), os.path.join(OUT_DIR, f"{word}_s{k}.png"))
        print(f"  {word}: 封面 + {min(3, len(files))} 张样本   共{len(files)}条样本")
    print(f"完成, 共 {total} 个词, 输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
