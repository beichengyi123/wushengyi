# -*- coding: utf-8 -*-
# ================= 手语实时识别脚本 =================
# 用法: 先跑 train_model.py 生成模型, 再运行本脚本
# 做的事:
#   1. 摄像头 + MediaPipe 提取手部关键点 (和采集脚本完全同一套归一化)
#   2. 凑满 45 帧窗口 -> 降采样 -> kNN 和模型里的样本比对 -> 得出词
#   3. 连续多帧认出同一个词才确认 (防抖), 间隔 2 秒才发一次 (防刷屏)
#   4. 通过 MQTT 发到 aiot/sign/gesture/demo, ESP32 屏幕显示
# 屏幕上会显示: 识别结果 / 置信距离(越小越像) / 采集样本数
# ====================================================

import cv2
import numpy as np
import mediapipe as mp
import os
import time
import paho.mqtt.client as mqtt
from PIL import Image, ImageDraw, ImageFont

# ---------------- 配置区 ----------------
MQTT_BROKER = "broker.emqx.io"
TOPIC_SIGN_TEXT = "aiot/sign/gesture/demo"   # 与 122.txt 一致, ESP32 订阅这个
K = 3              # kNN 邻居数
DIST_THRESHOLD = 3.0   # 最低门槛(兜底值), 实际用模型里每个词的自适应门槛
STABLE_NEED = 3        # 连续认对多少次窗口才确认 (3次≈0.2秒, 更灵敏)
PUBLISH_GAP = 1.5      # 两次发送最小间隔(秒)
FIXED_SEQ = 10         # 必须和 train_model.py 一致
RECORD_FRAMES = 45     # 必须和 sign_collect.py 一致
# ----------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sign_model.npz")

# ---------------- 中文字体 (opencv 自带字体画不了中文) ----------------
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 34)
small_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)

def draw_cn(frame, text, pos, ft=font, color=(0, 255, 0)):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img).text(pos, text, font=ft, fill=color)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ---------------- MediaPipe (与采集脚本一致) ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,          # 双手模式 (与采集脚本一致)
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

def frame_hands(result):
    """取一帧里两只手的原始坐标: 返回 (right_pts, left_pts), 缺哪只就是 None"""
    if not result.multi_hand_landmarks:
        return None
    right = left = None
    for i, hand_lm in enumerate(result.multi_hand_landmarks):
        label = result.multi_handedness[i].classification[0].label
        pts = np.array([[p.x, p.y, p.z] for p in hand_lm.landmark])
        if label == "Right":
            right = pts
        else:
            left = pts
    return (right, left)

def _normalize_seq(frames, anchor_pref):
    """按锚点偏好把帧序列组装成 (T,126) 特征 (与训练脚本同一套处理):
       锚点=偏好手(不在则用另一只), 降噪(B手以自身手腕为中心,相对位置降权0.4), 去z"""
    out = []
    for right, left in frames:
        anchor, other = (right, left) if anchor_pref == "Right" else (left, right)
        if anchor is None and other is None:
            return None
        if anchor is None:                 # 锚点手不在, 退回另一只
            anchor, other = other, None
        origin = anchor[0]
        scale = np.linalg.norm(anchor[9] - origin)
        if scale < 1e-6:
            return None
        a = ((anchor - origin) / scale).flatten()
        b = ((other - origin) / scale).flatten() if other is not None else np.zeros(63)
        out.append(np.concatenate([a, b]))
    arr = np.array(out, dtype=np.float32)
    b3 = arr[:, 63:].reshape(len(arr), 21, 3).copy()
    wrist = b3[:, 0:1, :]
    centered = b3 - wrist
    centered[:, 0, :] = wrist[:, 0, :] * 0.4
    arr = np.concatenate([arr[:, :63], centered.reshape(len(arr), 63)], axis=1)
    arr[:, 2::3] = 0
    return arr

def downsample(arr, n=FIXED_SEQ):
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return arr[idx].flatten()

MOTION_W = 0.7   # 必须和 train_model.py 一致

def finalize(arr):
    """静态轨迹 + 运动特征(帧间变化量), 与 train_model.py 完全一致"""
    ds = downsample(arr)
    seq = ds.reshape(FIXED_SEQ, -1)
    delta = np.zeros_like(seq)
    delta[1:] = seq[1:] - seq[:-1]
    return np.concatenate([ds, delta.flatten() * MOTION_W])

# ---------------- 加载模型 ----------------
if not os.path.exists(MODEL_PATH):
    print("找不到 sign_model.npz, 请先运行 train_model.py 训练")
    raise SystemExit(0)
model = np.load(MODEL_PATH, allow_pickle=False)
MX, My = model["X"], model["y"]
vocab = sorted(set(My.tolist()))
# 每个词的自适应门槛 (训练时按教材内离散程度算好存进模型的)
TH = dict(zip(model["thresh_words"].tolist(), model["thresh_vals"].tolist()))
# 每个词的运动指纹 (该词各维度平均运动量, 用于形状相近词的决胜)
DW = list(model["delta_words"].tolist())
DM = model["delta_matrix"]
print(f"模型加载成功: {len(My)} 条样本, 词库 {vocab}")
print("各词门槛: " + ", ".join(f"{w}:{v:.1f}" for w, v in TH.items()))

# ---------------- MQTT ----------------
client = mqtt.Client("PC_Sign_Recognizer")   # id 不能和 122.txt 重复
try:
    client.connect(MQTT_BROKER, 1883)
    client.loop_start()
    print(f"MQTT 已连接 {MQTT_BROKER}, 结果发往 {TOPIC_SIGN_TEXT}")
except Exception as e:
    print(f"MQTT 连接失败(不影响本地识别, 只是不发屏幕): {e}")

# ---------------- 主循环 ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("摄像头打开失败")
    raise SystemExit(0)

buffer = []          # 最近 45 帧特征
last_pred, last_dist = "...", None
stable_word, stable_cnt = None, 0
last_publish = 0
frame_no = 0

print("实时识别启动, 对镜头做手语, 按 q 退出\n")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS)
        fh = frame_hands(result)
        if fh is not None:
            buffer.append(fh)
            if len(buffer) > RECORD_FRAMES:
                buffer.pop(0)

            # 每 2 帧做一次分类 (更灵敏)
            frame_no += 1
            if len(buffer) == RECORD_FRAMES and frame_no % 2 == 0:
                # 两个锚点变体各算一遍, 取最近距离更小的 (防左右手标签翻转)
                best = None
                for pref in ("Right", "Left"):
                    arr = _normalize_seq(buffer, pref)
                    if arr is None:
                        continue
                    f = finalize(arr)   # 与训练一致: 静态轨迹+运动特征
                    d = np.linalg.norm(MX - f, axis=1)
                    seq = f[:840].reshape(FIXED_SEQ, -1)
                    live_move = np.abs(np.diff(seq, axis=0)).mean(axis=0)  # (84,) 直播运动指纹
                    if best is None or d.min() < best[2]:
                        best = (f, d, d.min(), live_move)
                if best is None:
                    continue
                f, d, _, live_move = best
                top = np.argsort(d)[:K]
                votes = [str(My[j]) for j in top]

                # 候选词去重
                uniq = []
                for v in votes:
                    if v not in uniq:
                        uniq.append(v)

                # 决胜规则: 票数接近时, 比较两个候选词在邻居里的平均距离,
                # 离哪团教材云更近就判谁 (实测比运动指纹更稳: 留一法 204/207, 零错认)
                def _md_of(w):
                    s = [d[j] for j in top if str(My[j]) == w]
                    return float(np.mean(s)) if s else 1e9

                if (len(uniq) >= 2
                        and votes.count(uniq[0]) - votes.count(uniq[1]) <= 1):
                    uniq.sort(key=_md_of)
                pred = uniq[0]
                same = [d[j] for j in top if str(My[j]) == pred]
                mean_d = float(np.mean(same)) if same else float(d[top[0]])
                th = float(TH.get(pred, DIST_THRESHOLD))

                if mean_d < th:
                    if pred == stable_word:
                        stable_cnt += 1
                    else:
                        stable_word, stable_cnt = pred, 1
                    last_pred, last_dist = pred, mean_d
                else:
                    last_pred, last_dist = "...", mean_d
                    stable_word, stable_cnt = None, 0

                # 防抖确认后发送
                if (stable_cnt == STABLE_NEED and stable_word
                        and time.time() - last_publish > PUBLISH_GAP):
                    client.publish(TOPIC_SIGN_TEXT, stable_word.encode("utf-8"))
                    print(f"[发送] {stable_word}  (dist={last_dist:.2f})")
                    last_publish = time.time()
                    stable_cnt = 0

    # ---- 界面 ----
    n_in = len(buffer)
    frame = draw_cn(frame, f"识别: {last_pred}", (180, 30),
                    color=(0, 255, 120) if last_pred != "..." else (200, 200, 200))
    if last_dist is not None:
        cur_th = float(TH.get(stable_word or last_pred, DIST_THRESHOLD)) if last_pred != "..." else DIST_THRESHOLD
        frame = draw_cn(frame, f"距离: {last_dist:.2f} (该词门槛{cur_th:.1f})", (180, 80), small_font, (160, 220, 255))
    if n_in < RECORD_FRAMES:
        frame = draw_cn(frame, f"手保持在画面里凑动作 {n_in}/{RECORD_FRAMES}", (180, 115), small_font, (0, 200, 255))
    cv2.imshow("Sign Recognizer", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
