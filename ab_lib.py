# -*- coding: utf-8 -*-
"""A/B 公用库: 加载「改造后」版本, 并把它的判定层可回放式地"降回"改造前, 用于回归对比。

重点说明: 「第一个版本.py」这个改造前的备份在我调试时被误覆盖了,
所以这里用 downgrade() 按原逻辑重建改造前的 LiveRecognizer(连续计数规则),
这样以后不依赖那个文件也能一直做 A/B 回归。
"""
import os
import time
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "sign_model.npz")
SRC_PATH = os.path.join(BASE, "1.py")

_m = np.load(MODEL_PATH, allow_pickle=False)
_tw = [str(w) for w in _m["thresh_words"].tolist()]
MODEL = {
    "MX": _m["X"].astype(np.float32), "My": _m["y"],
    "TH": dict(zip(_tw, _m["thresh_vals"].tolist())),
    "TH1": dict(zip(_tw, _m["thresh1_vals"].tolist())),
    "TY": dict(zip(_tw, _m["typical_vals"].tolist())),
    "TY1": dict(zip(_tw, _m["typical1_vals"].tolist())),
    "TWO": {str(w): bool(v) for w, v in zip(_tw, _m["two_hand_vals"].tolist())},
    "K": 3, "FIXED_SEQ": int(_m["fixed_seq"]), "MOTION_W": 0.7,
    "DIST_THRESHOLD": 4.0, "RECORD_FRAMES": 45, "MAX_SEG": 75,
    "SEG_DIM": int(_m["fixed_seq"]) * 126, "MIN_SEG": 10, "SEG_SILENCE": 4,
}
WORDS = sorted(set(str(v) for v in _m["y"].tolist()))


def build(path=SRC_PATH):
    """把 1.py 里「识别相关的代码块」抽出来跑一遍, 得到一个命名空间
    (跳过 MQTT/摄像头/GUI 那些带副作用的顶层代码)"""
    src = open(path, encoding="utf-8").read()
    b1 = src[src.index("def _normalize_seq("):src.index("# ================= 功能 1.5")]
    b2 = src[src.index("# ================= 功能 1a"):src.index("LIVE_WINS = ")]
    b3 = src[src.index("LIVE_WINS = "):src.index("def sign_language_thread(")]
    G = {"np": np, "time": time, "threading": __import__("threading")}
    G.update(MODEL)
    G["MY_S"] = np.array([str(v) for v in _m["y"].tolist()])
    exec(compile(b1, path, "exec"), G)
    exec(compile(b2, path, "exec"), G)
    G["WORD_IDX"] = {w: np.where(G["MY_S"] == w)[0] for w in np.unique(G["MY_S"])}
    exec(compile(b3, path, "exec"), G)
    return G


def downgrade(new_dict):
    """把新版命名空间里的判定层降回「改造前」那套逻辑, 用于 A/B 对比。

    改造前只有一条规则 —— 「同一个词连续认对 LIVE_NEED 次就出词」,
    没有滑窗多数表决 / 切换判定 / 按帧计算的出词间隔这三样东西。
    """
    G = dict(new_dict)
    LIVE_NEED, LIVE_RESET = 2, 3

    class LiveRecognizerOld:
        def __init__(self):
            self.buf = []
            self.frame_no = 0
            self.miss = 0
            self.no_word = 0
            self.pend, self.cnt = None, 0
            self.last_word = None
            self.pred, self.dist, self.th, self.one = "...", None, None, False

        def reset(self):
            self.buf.clear()
            self.pend, self.cnt = None, 0
            self.last_word = None
            self.no_word = 0

        def feed(self, fh):
            if fh is None:
                self.miss += 1
                self.no_word += 1
                if self.miss == 2:
                    self.buf.clear()
                    self.pend, self.cnt = None, 0
                if self.no_word >= LIVE_RESET:
                    self.last_word = None
                return None
            self.miss = 0
            self.buf.append(fh)
            if len(self.buf) > G["LIVE_BUF"]:
                self.buf.pop(0)
            self.frame_no += 1
            if len(self.buf) < G["LIVE_WINS"][0] or self.frame_no % 2:
                return None
            r = G["_live_classify"](self.buf)
            if r is None:
                return None
            sc, pred, dd, th, oh = r
            self.pred, self.dist, self.th, self.one = pred, dd, th, oh
            if dd >= th:
                self.no_word += 1
                self.pend, self.cnt = None, 0
                if self.no_word >= LIVE_RESET:
                    self.last_word = None
                return None
            self.no_word = 0
            if pred == self.pend:
                self.cnt += 1
            else:
                self.pend, self.cnt = pred, 1
            if self.cnt >= LIVE_NEED and pred != self.last_word:
                self.last_word = pred
                self.pend, self.cnt = None, 0
                return (pred, dd, th, oh)
            return None

    G["LiveRecognizer"] = LiveRecognizerOld
    G.pop("polish_words", None)      # 改造前没有词组合并层 -> polish() 会自动退化为原样输出
    return G


NEW = build()
OLD = downgrade(NEW)

POOL = {}
for w in WORDS:
    d = os.path.join(BASE, "sign_data", w)
    POOL[w] = [np.load(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".npy")]


def arr_to_frames(a):
    if a.shape[1] == 63:
        p = a.reshape(-1, 21, 3)
        return [(p[i].copy(), None) for i in range(len(a))]
    p = a[:, :63].reshape(-1, 21, 3)
    q = a[:, 63:].reshape(-1, 21, 3)
    w = q[:, 0:1, :] / 0.4
    q3 = q + w
    return [(p[i].copy(), None if not np.any(np.abs(q[i]) > 1e-6) else q3[i].copy())
            for i in range(len(a))]


def rs(seq, n):
    idx = np.linspace(0, len(seq) - 1, n).astype(int)
    return [seq[i] for i in idx]


def lcs(a, b):
    """最长公共子序列: 允许"多出/漏掉"词, 只统计认对且顺序对的词。
       直接按位置比对的话, 中间多一个词会让后面全部错位, 分数虚低。"""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = (dp[i][j] + 1 if a[i] == b[j]
                                else max(dp[i][j + 1], dp[i + 1][j]))
    return dp[-1][-1]


def polish(G, ws):
    """模拟上屏那一层: 词组合并纠错(改造前没有这层)"""
    if "polish_words" in G:
        return G["polish_words"]([[w, 0.4] for w in ws])
    return list(ws)


def run(G, frames):
    rec = G["LiveRecognizer"]()
    out = []
    for f in frames:
        r = rec.feed(f)
        if r:
            out.append(r[0])
    return out


def make_frames(seq, nper, gap, idx, rng, p_miss=0.0, sigma=0.0):
    """拼一段连续比划; p_miss=模拟 MediaPipe 偶尔漏检, sigma=手部关键点抖动强度"""
    frames = []
    for w in seq:
        for pt in rs(arr_to_frames(POOL[w][idx % len(POOL[w])]), nper):
            a, b = pt
            if rng.random() < p_miss:
                frames.append(None)          # 这一帧没检测到手
                continue
            if sigma > 0:
                a = a + rng.normal(0, sigma, a.shape).astype(np.float32)
                if b is not None:
                    b = b + rng.normal(0, sigma, b.shape).astype(np.float32)
            frames.append((a, b))
        if gap:
            frames += [None] * gap
    return frames


SENTS = [["我", "现在", "要", "去", "学校"],
         ["你", "好", "我", "叫", "名字"],
         ["我", "要", "吃饭"],
         ["现在", "去", "学校", "吃饭"],
         ["你好", "我", "叫", "名字"],
         ["学校", "什么", "现在"],
         ["我", "要", "帮助", "你"],       # 目标句: 帮助 重录后必测
         ["帮助", "学校", "名字"]]         # 三个双手词连打(历史互吃组合)
