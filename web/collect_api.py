# -*- coding: utf-8 -*-
"""
collect_api.py —— 手语样本采集 / AI 自检 / 入库（Flask 蓝图）

接进原有 5000 端口，不新增服务、不动原有路由。
在 web/app.py 里只加三行注册（见文件末尾注释）。

目录约定（和原有训练脚本 train_model.py 完全兼容）：
    正式动作库   sign_data/<标签>/<标签>_NNN_<时间戳>.npy      ← train_model.py 只读这里
    待考核样本库 sign_pending/<标签>/<标签>_NNN_<时间戳>.npy  ← 未通过考核的只留在这里
    每条样本旁边还有：
        <同名>.json  该条的手位统计（右手帧数/左手帧数/丢弃原因）
        <同名>.jpg   采集时的画面缩略图（网页预览用，训练脚本不读）
    标签与手位    web/data/sign_labels.json
        {"labels": {"学校": {"hand": "right", "status": "formal"}}}

样本格式与 sign_collect.py 一致：(T, 126) float32，每帧 = 右手63 + 左手63。
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np
from flask import Blueprint, Response, jsonify, render_template, request, send_file

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import sign_guard  # noqa: E402  防误检 + 手位规则（与 1.py 共用同一套）

DATA_DIR = os.path.join(BASE_DIR, "sign_data")            # 正式动作库
PENDING_DIR = os.path.join(BASE_DIR, "sign_pending")      # 待考核样本库
MODEL_PATH = os.path.join(BASE_DIR, "sign_model.npz")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train_model.py")

RECORD_FRAMES = 45          # 历史遗留: 旧版定长 45 帧。现在采集为"起止门控"变长样本, 该值仅作展示参考
MIN_VALID_FRAMES = 25       # 有效帧少于这个数就当采集失败（手没一直待在画面里）
                            # 45 帧里过 25 帧即可：训练时 down_sample 会等间隔取 10 帧，
                            # 少几帧不影响，但手得一直在画面里
DIM = 126
FIXED_SEQ = 10              # 与 train_model.py 一致

# ---- AI 自检的内部判定标准（页面上不暴露、也不让用户填数字）----
MIN_SAMPLES = 5             # 少于 5 条直接判"样本太少"
PASS_RATIO = 0.6            # 通过率低于 60% 判不合格
HAND_RATIO = 0.8            # 手位符合率低于 80% 判不合格
K = 3                       # kNN 邻居数，与 1.py / train_model.py 一致

bp = Blueprint("collect", __name__)

_hands = None
_model_cache = {"mtime": None, "data": None}
_mp_lock = threading.Lock()   # MediaPipe 实例不是线程安全的:
                              # 采集上传(45帧)和实时探测(probe)共用它, 逐帧上锁


# ===================== 基础工具 =====================
def _hand_detector():
    """独立的 MediaPipe Hands 实例（静态图片模式：每帧独立检测，适合拍照采集）"""
    global _hands
    if _hands is None:
        import mediapipe as mp
        _hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
    return _hands


def _trainer():
    """复用 train_model.py 里的 pad_frame / refine / finalize，
       保证"自检用的特征"和"训练/识别用的特征"是同一套，不重复造轮子"""
    import train_model
    return train_model


def _safe_label(label):
    label = (label or "").strip()
    if not label:
        raise ValueError("标签不能为空")
    if len(label) > 12:
        raise ValueError("标签最多 12 个字")
    bad = set('/\\:*?"<>|\n\r\t')
    if any(ch in bad for ch in label) or label.startswith("."):
        raise ValueError("标签里不能有 / \\ : * ? \" < > | 这些字符")
    return label


def _label_dir(root, label):
    d = os.path.join(root, label)
    os.makedirs(d, exist_ok=True)
    return d


def _count_npy(d):
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith(".npy")])


def _b64_to_bgr(data_url):
    """前端 canvas.toDataURL 的字符串 -> OpenCV BGR 图"""
    import cv2
    if not data_url:
        return None
    s = data_url.split(",", 1)[-1]
    raw = base64.b64decode(s)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _frame_features(bgr):
    """单帧 -> (126,) 特征 + 左右手坐标。人脸误检在这里被过滤掉。"""
    rgb = bgr[:, :, ::-1]                                # BGR -> RGB
    with _mp_lock:
        result = _hand_detector().process(rgb)
        boxes = sign_guard.face_boxes(rgb)
    right, left, dropped = sign_guard.pick_hands(result, boxes)
    anchor = right if right is not None else left
    if anchor is None:
        return None, None, None, dropped
    origin = anchor[0]
    scale = float(np.linalg.norm(anchor[9] - origin))
    if scale < 1e-6:
        return None, None, None, dropped
    a = ((anchor - origin) / scale).flatten()
    other = left if right is not None else right
    b = ((other - origin) / scale).flatten() if other is not None else np.zeros(63, dtype=np.float32)
    return np.concatenate([a, b]).astype(np.float32), right, left, dropped


def _new_prefix(d, label):
    n = _count_npy(d) + 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "%s_%03d_%s" % (label, n, stamp)


@bp.route("/api/collect/probe", methods=["POST"])
def api_probe():
    """实时探测一帧（不落盘）：采集页每隔几百毫秒发一小帧过来，
       返回手部 21 点归一化坐标、人脸过滤情况、画面亮度，
       前端据此画手部骨架、提示"没看到手 / 手贴脸 / 画面太暗"。

       探测用宽松门槛(0.5)：这里只负责"手在不在"的展示和起手门控，
       手刚伸进画面/角度偏时也能马上认出来，别让用户干等。
       真正存样本走 _frame_features 的严格门槛(0.75)，样本质量不打折。
       weak=True 表示手认出来了但没到严格档，提示用户靠近一点。"""
    d = request.get_json(force=True, silent=True) or {}
    bgr = _b64_to_bgr(d.get("frame") or "")
    if bgr is None:
        return jsonify(ok=False, msg="帧解码失败")
    import cv2
    bright = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())
    rgb = bgr[:, :, ::-1]
    with _mp_lock:
        result = _hand_detector().process(rgb)
        boxes = sign_guard.face_boxes(rgb)
        right, left, dropped = sign_guard.pick_hands(result, boxes, min_score=0.5)
        sright, sleft, _ = sign_guard.pick_hands(result, boxes)   # 严格档对照
    has_loose = right is not None or left is not None
    has_strict = sright is not None or sleft is not None
    weak = has_loose and not has_strict

    def pts(p):
        return None if p is None else [
            [round(float(q[0]), 4), round(float(q[1]), 4)] for q in p[:, :2]]

    return jsonify(ok=True, bright=round(bright, 1), weak=weak,
                   right=pts(right), left=pts(left),
                   dropped=[{"hand": h, "why": w} for h, w in dropped])


def _model():
    """读 sign_model.npz（带 mtime 缓存，重新训练后自动刷新）"""
    try:
        mtime = os.path.getmtime(MODEL_PATH)
    except OSError:
        return None
    if _model_cache["mtime"] == mtime and _model_cache["data"] is not None:
        return _model_cache["data"]
    try:
        z = np.load(MODEL_PATH, allow_pickle=False)
        data = {
            "X": z["X"], "y": z["y"],
            "words": [str(w) for w in z["thresh_words"].tolist()],
            "th": dict(zip([str(w) for w in z["thresh_words"].tolist()],
                           z["thresh_vals"].tolist())) if "thresh_vals" in z else {},
        }
        _model_cache["mtime"] = mtime
        _model_cache["data"] = data
        return data
    except Exception as e:
        print("[collect] 模型读取失败:", e)
        return None


def _vector(arr):
    """(T,126) 原始样本 -> 与 train_model.load_dataset 完全一致的特征向量"""
    tr = _trainer()
    arr = tr.pad_frame(np.asarray(arr, dtype=np.float32))
    if arr.shape[1] > DIM:
        arr = arr[:, :DIM]
    arr = arr.copy()
    arr[:, 2::3] = 0            # 去 z（和训练/识别一致）
    arr = tr.refine(arr)
    return tr.finalize(arr)


def _knn_predict(model, feat):
    """kNN 投票（与 1.py 同规则：票数优先）"""
    d = np.linalg.norm(model["X"] - feat, axis=1)
    top = np.argsort(d)[:K]
    votes = [str(model["y"][j]) for j in top]
    pred = max(set(votes), key=votes.count)
    dist = float(np.mean([d[j] for j in top if str(model["y"][j]) == pred]))
    return pred, dist, float(d.min())


def _sample_meta_path(npy_path):
    return os.path.splitext(npy_path)[0] + ".json"


def _load_sample_meta(npy_path):
    p = _sample_meta_path(npy_path)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            pass
    return {}


# ===================== 页面 =====================
@bp.route("/collect")
def collect_page():
    return render_template("collect.html", nav="collect")


# ===================== 标签 =====================
@bp.route("/api/collect/labels")
def api_labels():
    labels = sign_guard.load_labels(force=True)
    out = []
    for name, cfg in sorted(labels.items()):
        out.append({
            "label": name,
            "hand": cfg.get("hand", "none"),
            "hand_text": sign_guard.HAND_RULES.get(cfg.get("hand", "none"), "无手位限制"),
            "status": cfg.get("status", "pending"),
            "formal": _count_npy(os.path.join(DATA_DIR, name)),
            "pending": _count_npy(os.path.join(PENDING_DIR, name)),
        })
    return jsonify(ok=True, labels=out, rules=sign_guard.HAND_RULES,
                   min_samples=MIN_SAMPLES, record_frames=RECORD_FRAMES)


@bp.route("/api/collect/labels", methods=["POST"])
def api_create_label():
    d = request.get_json(force=True, silent=True) or {}
    try:
        label = _safe_label(d.get("label"))
    except ValueError as e:
        return jsonify(ok=False, msg=str(e)), 400
    hand = d.get("hand", "none")
    if hand not in sign_guard.HAND_RULES:
        return jsonify(ok=False, msg="手位只能是 right / left / none"), 400

    labels = dict(sign_guard.load_labels(force=True))
    if label in labels and (labels[label].get("status") == "formal" or
                            _count_npy(os.path.join(DATA_DIR, label)) > 0):
        return jsonify(ok=False, msg="「%s」已经在正式动作库里了，直接选它加样本即可" % label), 409

    labels[label] = {
        "hand": hand,
        "status": "pending",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    sign_guard.save_labels(labels)
    _label_dir(PENDING_DIR, label)
    return jsonify(ok=True, label=label, hand=hand)


@bp.route("/api/collect/labels/clear", methods=["POST"])
def api_clear_label():
    """清空某个标签的待考核样本（正式动作库不动，避免误删已入库数据）"""
    d = request.get_json(force=True, silent=True) or {}
    try:
        label = _safe_label(d.get("label"))
    except ValueError as e:
        return jsonify(ok=False, msg=str(e)), 400
    n = _delete_samples(os.path.join(PENDING_DIR, label))
    return jsonify(ok=True, label=label, removed=n)


def _delete_samples(d):
    n = 0
    if os.path.isdir(d):
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if os.path.isfile(p) and (f.endswith(".npy") or f.endswith(".json")
                                      or f.endswith(".jpg")):
                try:
                    os.remove(p)
                    n += 1 if f.endswith(".npy") else 0
                except OSError:
                    pass
    return n


# ===================== 样本 =====================
@bp.route("/api/collect/samples")
def api_samples():
    label = request.args.get("label", "")
    try:
        label = _safe_label(label)
    except ValueError as e:
        return jsonify(ok=False, msg=str(e)), 400
    d = os.path.join(PENDING_DIR, label)
    rows = []
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npy"):
                continue
            p = os.path.join(d, f)
            meta = _load_sample_meta(p)
            try:
                arr = np.load(p)
                frames = int(arr.shape[0])
            except Exception:
                frames = 0
            rows.append({
                "name": f,
                "frames": frames,
                "ts": meta.get("ts", ""),
                "right_frames": meta.get("right_frames", 0),
                "left_frames": meta.get("left_frames", 0),
                "dropped": meta.get("dropped", 0),
                "thumb": f[:-4] + ".jpg" if os.path.exists(p[:-4] + ".jpg") else "",
            })
    return jsonify(ok=True, label=label, samples=rows,
                   hand=sign_guard.get_hand_rule(label))


@bp.route("/api/collect/thumb/<label>/<name>")
def api_thumb(label, name):
    """样本缩略图（只允许读待考核库里的 jpg，防目录穿越）"""
    if "/" in name or "\\" in name or ".." in name or not name.endswith(".jpg"):
        return jsonify(ok=False, msg="非法文件名"), 400
    p = os.path.join(PENDING_DIR, os.path.basename(label), os.path.basename(name))
    if not os.path.exists(p):
        return jsonify(ok=False, msg="没有缩略图"), 404
    return send_file(p, mimetype="image/jpeg")


@bp.route("/api/collect/samples", methods=["POST"])
def api_upload_sample():
    """采集一条样本：前端把连续若干帧（base64 jpg）发上来，
       后端逐帧跑 MediaPipe 提特征，存成 (T,126) 的 .npy。"""
    d = request.get_json(force=True, silent=True) or {}
    try:
        label = _safe_label(d.get("label"))
    except ValueError as e:
        return jsonify(ok=False, msg=str(e)), 400
    frames = d.get("frames") or []
    if not isinstance(frames, list) or not frames:
        return jsonify(ok=False, msg="没有收到画面帧"), 400

    labels = sign_guard.load_labels(force=True)
    if label not in labels and not os.path.isdir(os.path.join(DATA_DIR, label)):
        return jsonify(ok=False, msg="标签不存在，请先新增标签"), 404

    rule = sign_guard.get_hand_rule(label)
    feats, right_n, left_n, dropped_n, thumb = [], 0, 0, 0, None
    bad_decode = no_hand = drop_face = drop_score = 0
    for f in frames:
        bgr = _b64_to_bgr(f)
        if bgr is None:
            bad_decode += 1
            continue
        vec, right, left, dropped = _frame_features(bgr)
        dropped_n += len(dropped)
        for _lab, why in dropped:
            if why == "inside_face":
                drop_face += 1
            elif why == "low_score":
                drop_score += 1
        if vec is None:
            no_hand += 1
            continue
        feats.append(vec)
        if right is not None:
            right_n += 1
        if left is not None:
            left_n += 1
        if thumb is None:
            thumb = bgr

    if len(feats) < MIN_VALID_FRAMES:
        # 把"为什么不够"讲清楚：光说帧数不够，用户不知道该改什么
        parts = ["只采到 %d 帧有手（最少要 %d 帧）" % (len(feats), MIN_VALID_FRAMES)]
        det = []
        if no_hand:
            det.append("%d 帧完全没检测到手" % no_hand)
        if drop_face:
            det.append("%d 帧的手判定为贴在脸上被过滤" % drop_face)
        if drop_score:
            det.append("%d 帧手部置信度太低" % drop_score)
        if bad_decode:
            det.append("%d 帧图像解码失败" % bad_decode)
        if det:
            parts.append("（" + "、".join(det) + "）")
        if drop_face >= 5 and drop_face >= no_hand:
            parts.append("。手别贴着脸，往画面中间、下巴以下的位置放。")
        elif no_hand >= 5:
            parts.append("。手要整只放进画面中间，别只露指尖，也别离镜头太近或太远。")
        else:
            parts.append("。按采集后把手举进画面，完整做一遍动作，做完停一下等它自动结束。")
        return jsonify(ok=False, msg="".join(parts),
                       diagnosis={"total": len(frames), "valid": len(feats),
                                  "no_hand": no_hand, "inside_face": drop_face,
                                  "low_score": drop_score, "bad_decode": bad_decode}), 400

    # 手位当场校验：采集阶段就按登记的手位要求把关，不合格的样本不浪费磁盘
    if rule == "right" and right_n < len(feats) * 0.6:
        return jsonify(ok=False, msg="这条登记为「右手」，但画面里右手只出现了 %d/%d 帧。"
                                     "请用右手重做。" % (right_n, len(feats))), 400
    if rule == "left" and left_n < len(feats) * 0.6:
        return jsonify(ok=False, msg="这条登记为「左手」，但画面里左手只出现了 %d/%d 帧。"
                                     "请用左手重做。" % (left_n, len(feats))), 400

    dlabel = _label_dir(PENDING_DIR, label)
    prefix = _new_prefix(dlabel, label)
    npy_path = os.path.join(dlabel, prefix + ".npy")
    np.save(npy_path, np.array(feats, dtype=np.float32))

    meta = {
        "label": label,
        "hand_rule": rule,
        "frames": len(feats),
        "right_frames": right_n,
        "left_frames": left_n,
        "dropped": dropped_n,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "web-collect",
    }
    with open(_sample_meta_path(npy_path), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    if thumb is not None:
        import cv2
        ok, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            open(npy_path[:-4] + ".jpg", "wb").write(buf.tobytes())

    return jsonify(ok=True, name=os.path.basename(npy_path), frames=len(feats),
                   right_frames=right_n, left_frames=left_n,
                   dropped=dropped_n, total=_count_npy(dlabel))


@bp.route("/api/collect/samples", methods=["DELETE"])
def api_delete_sample():
    d = request.get_json(force=True, silent=True) or {}
    try:
        label = _safe_label(d.get("label"))
    except ValueError as e:
        return jsonify(ok=False, msg=str(e)), 400
    name = os.path.basename(d.get("name") or "")
    if not name.endswith(".npy"):
        return jsonify(ok=False, msg="非法文件名"), 400
    base = os.path.join(PENDING_DIR, label, name[:-4])
    removed = False
    for ext in (".npy", ".json", ".jpg"):
        p = base + ext
        if os.path.exists(p):
            try:
                os.remove(p)
                removed = True
            except OSError:
                pass
    if not removed:
        return jsonify(ok=False, msg="样本不存在"), 404
    return jsonify(ok=True, total=_count_npy(os.path.join(PENDING_DIR, label)))


# ===================== AI 自检考核 =====================
def _check_one(model, label, npy_path, meta, rule):
    """单条样本的双重校验：① 本地图像模型识别结果 ② 手位是否符合登记规则"""
    detail = {"name": os.path.basename(npy_path), "model": {}, "hand": {}, "ok": False}
    try:
        arr = np.load(npy_path)
    except Exception as e:
        detail["reason"] = "样本读不出来: %s" % e
        return detail

    frames = int(arr.shape[0])
    # ---- ② 手位校验（用采集时记录的真实左右手出现帧数）----
    rf = int(meta.get("right_frames", 0))
    lf = int(meta.get("left_frames", 0))
    if rule == "right":
        hand_ok = rf >= frames * 0.6 and rf > 0
    elif rule == "left":
        hand_ok = lf >= frames * 0.6 and lf > 0
    else:
        hand_ok = (rf + lf) > 0
    detail["hand"] = {"rule": rule, "right_frames": rf, "left_frames": lf,
                      "frames": frames, "ok": bool(hand_ok)}

    # ---- ① 图像模型识别 ----
    feat = _vector(arr)
    if model is None:
        detail["model"] = {"ok": False, "reason": "本地模型文件缺失(sign_model.npz)"}
        detail["reason"] = "本地模型不可用"
        return detail

    pred, dist, nearest = _knn_predict(model, feat)
    th = float(model["th"].get(label, 0) or 0)
    in_model = label in model["words"]

    if in_model:
        model_ok = (pred == label) and (th <= 0 or dist <= th * 1.15)
        detail["model"] = {"pred": pred, "dist": round(dist, 2), "th": round(th, 2),
                           "nearest": round(nearest, 2), "mode": "model", "ok": bool(model_ok)}
    else:
        # 新标签还没进模型：只能做"别和已有词撞车"的冲突检查
        conflict = nearest < 0.75 * dist if dist > 1e-6 else False
        detail["model"] = {"pred": pred, "dist": round(dist, 2), "th": round(th, 2),
                           "nearest": round(nearest, 2), "mode": "conflict-check",
                           "ok": not conflict}
        model_ok = not conflict

    detail["ok"] = bool(model_ok and hand_ok)
    if not model_ok:
        detail["reason"] = "模型识别成「%s」" % detail["model"].get("pred", "?")
    elif not hand_ok:
        detail["reason"] = "手位不符合登记规则(%s)" % sign_guard.HAND_RULES.get(rule, rule)
    else:
        detail["reason"] = "通过"
    return detail


@bp.route("/api/collect/selfcheck", methods=["POST"])
def api_selfcheck():
    """AI 自检：只用本地模型 + 手位规则，不联网、不调外部 API"""
    d = request.get_json(force=True, silent=True) or {}
    try:
        label = _safe_label(d.get("label"))
    except ValueError as e:
        return jsonify(ok=False, msg=str(e)), 400

    dlabel = os.path.join(PENDING_DIR, label)
    files = sorted(f for f in os.listdir(dlabel) if f.endswith(".npy")) \
        if os.path.isdir(dlabel) else []
    rule = sign_guard.get_hand_rule(label)

    if len(files) < MIN_SAMPLES:
        return jsonify(ok=True, verdict="fail", label=label, hand=rule,
                       samples=len(files), score=0,
                       reason="待考核样本只有 %d 条，至少 %d 条才有统计意义。"
                              % (len(files), MIN_SAMPLES),
                       details=[],
                       advice=["继续采集：换个角度、换点距离、换只手的摆放高低，"
                               "每种变化都录一两条（同一姿势重复 10 遍和 1 遍没区别）。",
                               "如果动作本身有问题，可以清空待考核样本重新采集。"])

    model = _model()
    details = []
    for f in files:
        p = os.path.join(dlabel, f)
        details.append(_check_one(model, label, p, _load_sample_meta(p), rule))

    ok_n = sum(1 for x in details if x["ok"])
    score = ok_n / len(details)
    hand_n = sum(1 for x in details if x["hand"].get("ok"))
    model_n = sum(1 for x in details if x["model"].get("ok"))
    passed = (score >= PASS_RATIO) and (hand_n / len(details) >= HAND_RATIO)

    # 给一句像人说的结论，而不是光秃秃的百分比
    if passed:
        reason = ("%d 条样本里 %d 条通过（模型 %d 条、手位 %d 条），" 
                  "规律一致，可以入库。" % (len(details), ok_n, model_n, hand_n))
        advice = []
    else:
        wrong = {}
        for x in details:
            if not x["ok"]:
                pred = x["model"].get("pred")
                key = ("手位不对" if not x["hand"].get("ok") and x["model"].get("ok")
                       else ("被认成「%s」" % pred if pred and pred != label else "不稳定"))
                wrong[key] = wrong.get(key, 0) + 1
        top = max(wrong.items(), key=lambda kv: kv[1])[0] if wrong else "不稳定"
        reason = ("%d 条样本只有 %d 条通过（模型 %d 条、手位 %d 条），主要问题：%s。"
                  % (len(details), ok_n, model_n, hand_n, top))
        advice = [
            "A · 继续补样本：换角度（正面/侧面 30°）、换光照（开灯/背光）、"
            "换手的远近高低，每条都按登记的手位比；补 5~10 条再自检一次。",
            "B · 清空重采：如果这个动作本身和已有词太像（比如都像「好」），"
            "把待考核样本全部删掉，改用差别更大的手型重录。",
        ]

    return jsonify(ok=True, verdict="pass" if passed else "fail", label=label,
                   hand=rule, hand_text=sign_guard.HAND_RULES.get(rule, ""),
                   samples=len(details), score=round(score, 3),
                   hand_pass=round(hand_n / len(details), 3),
                   model_pass=round(model_n / len(details), 3),
                   reason=reason, advice=advice,
                   details=details[:60])


@bp.route("/api/collect/promote", methods=["POST"])
def api_promote():
    """确认入库：把待考核样本连同手位配置一起移入正式动作库 sign_data/<标签>/"""
    d = request.get_json(force=True, silent=True) or {}
    try:
        label = _safe_label(d.get("label"))
    except ValueError as e:
        return jsonify(ok=False, msg=str(e)), 400

    src = os.path.join(PENDING_DIR, label)
    files = [f for f in os.listdir(src) if f.endswith(".npy")] if os.path.isdir(src) else []
    if not files:
        return jsonify(ok=False, msg="没有待入库的样本"), 404

    dst = _label_dir(DATA_DIR, label)
    moved = 0
    for f in files:
        base = f[:-4]
        for ext in (".npy", ".json", ".jpg"):
            p = os.path.join(src, base + ext)
            if not os.path.exists(p):
                continue
            target = os.path.join(dst, os.path.basename(p))
            if os.path.exists(target):      # 理论上不会重名，防一手
                target = os.path.join(dst, base + "_r" + ext)
            shutil.move(p, target)
            if ext == ".npy":
                moved += 1

    # 手位配置跟着标签一起转正
    labels = dict(sign_guard.load_labels(force=True))
    cfg = dict(labels.get(label) or {"hand": "none", "created": ""})
    cfg["status"] = "formal"
    cfg["promoted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    labels[label] = cfg
    sign_guard.save_labels(labels)

    return jsonify(ok=True, label=label, moved=moved,
                   formal=_count_npy(dst), hand=cfg.get("hand", "none"))


@bp.route("/api/collect/retrain", methods=["POST"])
def api_retrain():
    """入库后重训本地模型（可选）。训练脚本没改，直接调用原来的 train_model.py"""
    if not os.path.exists(TRAIN_SCRIPT):
        return jsonify(ok=False, msg="找不到 train_model.py"), 404
    try:
        # 不指定 text/encoding: 拿原始字节自己解。Windows 上子进程中文输出
        # 默认走 GBK, 而 PYTHONIOENCODING=utf-8 时又走 UTF-8, 写死哪个都会乱码
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, TRAIN_SCRIPT], cwd=BASE_DIR,
                           capture_output=True, timeout=300, env=env)
        raw = (r.stdout or b"") + b"\n" + (r.stderr or b"")
        text = None
        for enc in ("utf-8", "gbk"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = raw.decode("utf-8", errors="replace")
        tail = [ln for ln in text.strip().splitlines() if ln.strip()][-12:]
        return jsonify(ok=(r.returncode == 0), code=r.returncode, log=tail,
                       msg="模型已更新，重启识别程序后生效" if r.returncode == 0 else "训练失败")
    except subprocess.TimeoutExpired:
        return jsonify(ok=False, msg="训练超时（样本太多？）"), 504
    except Exception as e:
        return jsonify(ok=False, msg=str(e)), 500


# ===================== 直播围观（手机扫码同步看电脑端采集画面）=====================
# 电脑采集页(本机模式)每 ~0.2s 把"画面+骨架+状态"推到 push;
# 手机端(围观模式)通过 frame/state 拉取, 实现多人同步围观采集全过程。
# 后端只做内存中转(存最新一帧), 不落盘、不占摄像头。
_mirror = {"jpeg": None, "ts": 0.0, "state": {}, "lock": threading.Lock()}


@bp.route("/api/collect/mirror/push", methods=["POST"])
def api_mirror_push():
    """电脑采集页推一帧（JSON: {frame: dataURL, state: {...}}）"""
    d = request.get_json(force=True, silent=True) or {}
    b64 = (d.get("frame") or "").split(",", 1)[-1]
    try:
        jpeg = base64.b64decode(b64) if b64 else b""
    except Exception:
        jpeg = b""
    with _mirror["lock"]:
        if jpeg:
            _mirror["jpeg"] = jpeg
            _mirror["ts"] = time.time()
        if isinstance(d.get("state"), dict):
            _mirror["state"] = d["state"]
    return jsonify(ok=True)


@bp.route("/api/collect/mirror/frame")
def api_mirror_frame():
    """围观端拉最新一帧 JPEG（img 轮询方式，微信内置浏览器也兼容）"""
    with _mirror["lock"]:
        jpeg = _mirror["jpeg"]
    if not jpeg:
        return jsonify(ok=False, msg="电脑端还没开始采集直播"), 404
    resp = Response(jpeg, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@bp.route("/api/collect/mirror/state")
def api_mirror_state():
    """围观端轮询：电脑端是否在线 + 当前采集状态"""
    now = time.time()
    with _mirror["lock"]:
        ts = _mirror["ts"]
        state = dict(_mirror["state"])
    return jsonify(ok=True,
                   live=bool(now - ts < 3),          # 3 秒内有推帧 = 直播中
                   age=round(now - ts, 1), state=state)


# ===================== 采集页在线心跳（给桌面软件让摄像头用） =====================
_collect_last_seen = 0.0     # 采集页最近一次心跳时间


@bp.route("/api/collect/heartbeat", methods=["POST"])
def api_heartbeat():
    """采集页打开期间周期性调用。桌面软件看到"采集页在线"就主动让出摄像头"""
    global _collect_last_seen
    _collect_last_seen = time.time()
    return jsonify(ok=True)


@bp.route("/api/collect/busy", methods=["GET"])
def api_collect_busy():
    """采集页是否在线(8 秒内有心跳算在线)。桌面软件轮询这个决定摄像头归属"""
    return jsonify(ok=True, busy=bool(time.time() - _collect_last_seen < 8))


# ===================== 注册方式 =====================
# 在 web/app.py 里（其它 @app.route 之后、if __name__ == "__main__" 之前）加：
#     from collect_api import bp as collect_bp
#     app.register_blueprint(collect_bp)
# 然后访问 http://<本机IP>:5000/collect
