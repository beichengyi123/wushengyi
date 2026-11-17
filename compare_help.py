# -*- coding: utf-8 -*-
"""对照两个模型在同一个句子集上的表现(专门看「帮助」重录前后的差别)。

用法: python compare_help.py
默认比 sign_model_bak_帮助重录前.npz (旧) 和 sign_model.npz (新)。
想比别的模型就改下面 OLD / NEW。
"""
import os
import time
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(BASE, "1.py"), encoding="utf-8").read()

OLD = os.path.join(BASE, "sign_model_bak_帮助重录前.npz")
NEW = os.path.join(BASE, "sign_model.npz")


def build(model_path):
    """从 1.py 抽出识别相关代码块, 用指定模型装配出一个可回放的命名空间"""
    m = np.load(model_path, allow_pickle=False)
    words = [str(w) for w in m["thresh_words"].tolist()]
    b1 = SRC[SRC.index("def _normalize_seq("):SRC.index("# ================= 功能 1.5")]
    b2 = SRC[SRC.index("# ================= 功能 1a"):SRC.index("LIVE_WINS = ")]
    b3 = SRC[SRC.index("LIVE_WINS = "):SRC.index("def sign_language_thread(")]
    G = {"np": np, "time": time, "threading": __import__("threading"),
         "MX": m["X"].astype(np.float32), "My": m["y"],
         "TH": dict(zip(words, m["thresh_vals"].tolist())),
         "TH1": dict(zip(words, m["thresh1_vals"].tolist())),
         "TY": dict(zip(words, m["typical_vals"].tolist())),
         "TY1": dict(zip(words, m["typical1_vals"].tolist())),
         "TWO": {w: bool(v) for w, v in zip(words, m["two_hand_vals"].tolist())},
         "MY_S": np.array([str(v) for v in m["y"].tolist()]),
         "K": 3, "FIXED_SEQ": int(m["fixed_seq"]), "MOTION_W": 0.7,
         "DIST_THRESHOLD": 4.0, "RECORD_FRAMES": 45, "MAX_SEG": 75,
         "SEG_DIM": int(m["fixed_seq"]) * 126, "MIN_SEG": 10, "SEG_SILENCE": 4}
    exec(compile(b1, "1.py", "exec"), G)
    exec(compile(b2, "1.py", "exec"), G)
    G["WORD_IDX"] = {w: np.where(G["MY_S"] == w)[0] for w in np.unique(G["MY_S"])}
    exec(compile(b3, "1.py", "exec"), G)
    return G, words


def arr_to_frames(a):
    """(T,126) 样本 -> LiveRecognizer 吃的 [(右手21点, 左手21点/None), ...]
       (把训练时降权过的第二只手腕位置还原回去)"""
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
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = (dp[i][j] + 1 if a[i] == b[j]
                                else max(dp[i][j + 1], dp[i + 1][j]))
    return dp[-1][-1]


POOL = {}
for w in os.listdir(os.path.join(BASE, "sign_data")):
    d = os.path.join(BASE, "sign_data", w)
    if os.path.isdir(d):
        POOL[w] = [arr_to_frames(np.load(os.path.join(d, f)))
                   for f in sorted(os.listdir(d)) if f.endswith(".npy")]

# 重点句子: 帮助 单独 / 我要帮助你 / 帮助和学校挨着(历史互吃组合)
SENTS = [["我", "要", "帮助", "你"],
         ["帮助"],
         ["帮助", "你"],
         ["我", "现在", "要", "去", "学校"],
         ["学校", "帮助", "你"],
         ["我", "要", "帮助", "吃饭"]]


def run(G, seq, nper, gap, idx):
    frames = []
    for w in seq:
        frames += rs(POOL[w][idx % len(POOL[w])], nper)
        if gap:
            frames += [None] * gap
    rec = G["LiveRecognizer"]()
    out = []
    for f in frames:
        r = rec.feed(f)
        if r:
            out.append(r[0])
    pol = G["polish_words"]([[w, 0.4] for w in out]) if "polish_words" in G else out
    return pol


def report(tag, path):
    G, words = build(path)
    print(f"\n===== {tag}  ({os.path.basename(path)}) =====")
    print("  门槛: " + ", ".join(f"{w}={G['TH'][w]:.1f}" for w in ("帮助", "学校", "名字")))
    so = sn = hit = tot = 0
    for nper in (20, 30, 45):
        for gap in (0, 5):
            for si, seq in enumerate(SENTS):
                pol = run(G, seq, nper, gap, 5 + si)
                so += (pol == seq)
                hit += lcs(seq, pol)
                sn += 1
                tot += len(seq)
        # 每种节奏单独看
    print(f"  整句全对 {so}/{sn} = {so/sn*100:.0f}%   词级(LCS) {hit/tot*100:.1f}%")
    print("  --- 逐句(30帧/词, 无停顿) ---")
    for si, seq in enumerate(SENTS):
        pol = run(G, seq, 30, 0, 5 + si)
        ok = "✓" if pol == seq else "✗"
        print(f"   {ok} 真值 {' '.join(seq):<20} 出的 {' '.join(pol):<20}")


report("重录前(旧样本)", OLD)
report("重录后(新样本)", NEW)
