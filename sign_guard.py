# -*- coding: utf-8 -*-
"""
sign_guard.py —— 手语识别/采集共用的"守卫层"

做两件事（不碰原有识别、推送、TFT 逻辑，只在外面加两道闸）：

  1) 防误检：MediaPipe 有时会把脸（下巴/嘴角/耳朵）识别成手部关键点，
     于是"人脸"被当成手势参与推理，凭空多出一个词。
     这里做两条过滤：
       · 手部置信下限（handedness score 太低的不算手）
       · 落在人脸框里的手骨架直接丢弃
  2) 手位校验：每个标签可以登记"左手 / 右手 / 无限制"，
     出词时按登记的手位核对，外观近似但手位不同的动作不会被认成同一个词。

被三处复用：
  · 1.py         实时识别（frame_hands / LiveRecognizer）
  · gui/gui_app  桌面端摄像头线程
  · web/collect_api.py  采集页与 AI 自检

标签配置存放：web/data/sign_labels.json
  {"labels": {"学校": {"hand": "right", "status": "formal", ...}}}
不改这个文件也不影响老功能：没登记的标签一律按"无手位限制"处理。
"""

import json
import os
import threading
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(BASE_DIR, "web", "data", "sign_labels.json")

# ---- 可调参数（都集中在这里，改完重启服务/程序生效）----
MIN_HAND_SCORE = 0.75      # 手部置信下限：handedness 分数低于它的手直接不当手
FACE_INSIDE_RATIO = 0.88   # 一只手有这么大比例的关键点落在人脸框里 -> 判为人脸误检
FACE_BOX_PAD = 0.05        # 人脸框外扩一点，容忍下巴/刘海边缘
FACE_MIN_CONF = 0.5        # 人脸检测自身置信度
FACE_EVERY = 2             # 每 N 帧跑一次人脸检测（人脸位置变化慢，省算力）
#
# 注意(踩过的坑)：这两个值最早是 0.5 / 0.12，结果手语比划时手经常在脸前、
# 脸旁，手掌一大半关键点落进外扩后的脸框 -> 真手被当成"人脸误检"丢掉，
# 采集时有效帧不够、屏幕上也不出词。真实的人脸被误识别成手时，21 个关键点
# 几乎全在脸框内(比例≈1.0)，所以收到 0.88 仍然挡得住，同时不会误杀贴脸的手势。


HAND_RULES = {"right": "右手", "left": "左手", "none": "无手位限制"}

_lock = threading.Lock()
_cache = {"mtime": None, "labels": {}}
_face_det = None


# ===================== 一、标签 / 手位配置 =====================
def default_labels():
    """第一次运行时用 sign_data 里已有的词初始化，全部按"无限制"处理，
       保证老词库行为完全不变"""
    out = {}
    data_dir = os.path.join(BASE_DIR, "sign_data")
    if os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            if os.path.isdir(os.path.join(data_dir, name)):
                out[name] = {"hand": "none", "status": "formal", "created": ""}
    return out


def load_labels(force=False):
    """读标签配置（带 mtime 缓存，别的进程改了也能自动刷新）"""
    with _lock:
        try:
            mtime = os.path.getmtime(LABELS_PATH)
        except OSError:
            mtime = None
        if not force and _cache["mtime"] == mtime and _cache["labels"]:
            return _cache["labels"]
        labels = {}
        if mtime is not None:
            try:
                with open(LABELS_PATH, "r", encoding="utf-8") as fp:
                    labels = (json.load(fp) or {}).get("labels", {})
            except Exception as e:
                print("[sign_guard] 标签配置读取失败:", e)
        if not labels:
            labels = default_labels()
        # 兜底：手位只认三个值
        for name, cfg in labels.items():
            if not isinstance(cfg, dict):
                labels[name] = {"hand": "none", "status": "pending", "created": ""}
            elif cfg.get("hand") not in HAND_RULES:
                cfg["hand"] = "none"
        _cache["mtime"] = mtime
        _cache["labels"] = labels
        return labels


def save_labels(labels):
    os.makedirs(os.path.dirname(LABELS_PATH), exist_ok=True)
    tmp = LABELS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump({"labels": labels}, fp, ensure_ascii=False, indent=2)
    os.replace(tmp, LABELS_PATH)
    with _lock:
        _cache["mtime"] = None
        _cache["labels"] = {}


def get_hand_rule(word):
    cfg = load_labels().get(word) or {}
    rule = cfg.get("hand", "none")
    return rule if rule in HAND_RULES else "none"


def set_hand_rule(word, rule):
    if rule not in HAND_RULES:
        raise ValueError("手位只能是 right / left / none")
    labels = dict(load_labels(force=True))
    cfg = dict(labels.get(word) or {"status": "pending", "created": ""})
    cfg["hand"] = rule
    labels[word] = cfg
    save_labels(labels)


def hand_rule_ok(word, has_right, has_left):
    """手位校验：这个词登记的是哪只手，画面里就得真有那只手。
       none / 未登记 -> 永远通过（不影响老词）"""
    rule = get_hand_rule(word)
    if rule == "right":
        return bool(has_right)
    if rule == "left":
        return bool(has_left)
    return True


def hand_rule_reason(word, has_right, has_left):
    """不通过时给一句人话，方便日志/页面提示"""
    rule = get_hand_rule(word)
    if rule == "right" and not has_right:
        return "「%s」登记为右手，但画面里没有右手" % word
    if rule == "left" and not has_left:
        return "「%s」登记为左手，但画面里没有左手" % word
    return ""


# ===================== 二、人脸误检过滤 =====================
def _detector():
    global _face_det
    if _face_det is None:
        import mediapipe as mp
        _face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=FACE_MIN_CONF)
    return _face_det


class FaceGuard:
    """按帧间隔跑人脸检测，缓存上一次的人脸框（归一化坐标 x1,y1,x2,y2）"""

    def __init__(self, every=FACE_EVERY):
        self.every = max(1, int(every))
        self.n = 0
        self.boxes = []

    def boxes_for(self, rgb):
        self.n += 1
        if rgb is not None and (self.n % self.every == 1 or not self.boxes):
            try:
                res = _detector().process(rgb)
                boxes = []
                for det in (res.detections or []):
                    b = det.location_data.relative_bounding_box
                    x1, y1 = b.xmin, b.ymin
                    x2, y2 = b.xmin + b.width, b.ymin + b.height
                    px = (x2 - x1) * FACE_BOX_PAD
                    py = (y2 - y1) * FACE_BOX_PAD
                    boxes.append((x1 - px, y1 - py, x2 + px, y2 + py))
                self.boxes = boxes
            except Exception:
                self.boxes = []
        return self.boxes


_face_guard = FaceGuard()


def face_boxes(rgb):
    """给一帧 rgb 拿到人脸框（归一化坐标）；rgb 为 None 时返回空"""
    if rgb is None:
        return []
    return _face_guard.boxes_for(rgb)


def _inside_face(pts, boxes):
    """这只手有多大比例的关键点落在人脸框内"""
    if not boxes or pts is None:
        return 0.0
    xy = pts[:, :2]
    best = 0.0
    for (x1, y1, x2, y2) in boxes:
        inside = ((xy[:, 0] >= x1) & (xy[:, 0] <= x2) &
                  (xy[:, 1] >= y1) & (xy[:, 1] <= y2)).mean()
        best = max(best, float(inside))
    return best


def pick_hands(result, boxes=None, min_score=MIN_HAND_SCORE):
    """把 MediaPipe Hands 的结果清成 (right, left, dropped)
       right/left: (21,3) 数组或 None
       dropped: [("Right", "low_score"/"inside_face"), ...] 仅用于日志排查"""
    right = left = None
    dropped = []
    if result is None or not getattr(result, "multi_hand_landmarks", None):
        return None, None, dropped

    for i, hand_lm in enumerate(result.multi_hand_landmarks):
        label, score = "Right", 1.0
        try:
            cls = result.multi_handedness[i].classification[0]
            label, score = cls.label, float(cls.score)
        except Exception:
            pass
        pts = np.array([[p.x, p.y, p.z] for p in hand_lm.landmark], dtype=np.float32)

        # 闸 1：置信下限
        if score < min_score:
            dropped.append((label, "low_score"))
            continue
        # 闸 2：人脸区域里的手骨架
        if _inside_face(pts, boxes) >= FACE_INSIDE_RATIO:
            dropped.append((label, "inside_face"))
            continue

        if label == "Right":
            # 同一只手被检出两次时保留置信度更高的
            if right is None or score >= 0:
                right = pts
        else:
            left = pts
    return right, left, dropped


def has_hands(result, boxes=None, min_score=MIN_HAND_SCORE):
    """只要判断"有没有可用手"，不需要坐标时用这个"""
    r, l, _ = pick_hands(result, boxes, min_score)
    return (r is not None) or (l is not None)
