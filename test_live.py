# -*- coding: utf-8 -*-
# 离线验收: 实时出词(LiveRecognizer) —— 每词多快、多少个词、延迟多少
import os, time
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(BASE, "1.py"), encoding="utf-8").read()
b1 = SRC[SRC.index("def _normalize_seq("):SRC.index("# ================= 功能 1.5")]
b2 = SRC[SRC.index("# ================= 功能 1a"):SRC.index("LIVE_WINS = ")]
b3 = SRC[SRC.index("LIVE_WINS = "):SRC.index("def sign_language_thread(")]

m = np.load(os.path.join(BASE, "sign_model.npz"), allow_pickle=False)
X = m["X"].astype(np.float32); My = m["y"]
words_all = [str(w) for w in m["thresh_words"].tolist()]
G = {"np": np, "time": time, "threading": __import__("threading"),
     "MX": X, "My": My,
     "TH": dict(zip(words_all, m["thresh_vals"].tolist())),
     "TH1": dict(zip(words_all, m["thresh1_vals"].tolist())),
     "TY": dict(zip(words_all, m["typical_vals"].tolist())),
     "TY1": dict(zip(words_all, m["typical1_vals"].tolist())) if "typical1_vals" in m
            else dict(zip(words_all, m["typical_vals"].tolist())),
     "TWO": {w: bool(v) for w, v in zip(words_all, m["two_hand_vals"].tolist())},
     "MY_S": np.array([str(v) for v in My.tolist()]),
     "K": 3, "FIXED_SEQ": int(m["fixed_seq"]), "MOTION_W": 0.7,
     "DIST_THRESHOLD": 4.0, "RECORD_FRAMES": 45, "MAX_SEG": 75,
     "SEG_DIM": int(m["fixed_seq"]) * 126,
     "MIN_SEG": 10, "SEG_SILENCE": 4}
exec(compile(b1, "1.py", "exec"), G)
exec(compile(b2, "1.py", "exec"), G)
G["WORD_IDX"] = {w: np.where(G["MY_S"] == w)[0] for w in np.unique(G["MY_S"])}
exec(compile(b3, "1.py", "exec"), G)
Live = G["LiveRecognizer"]


def arr_to_frames(a):
    if a.shape[1] == 63:
        p = a.reshape(-1, 21, 3)
        return [(p[i].copy(), None) for i in range(len(a))]
    p = a[:, :63].reshape(-1, 21, 3); q = a[:, 63:].reshape(-1, 21, 3)
    w = q[:, 0:1, :] / 0.4; q3 = q + w
    return [(p[i].copy(), None if not np.any(np.abs(q[i]) > 1e-6) else q3[i].copy())
            for i in range(len(a))]


POOL = {}
for w in words_all:
    d = os.path.join(BASE, "sign_data", w)
    POOL[w] = [arr_to_frames(np.load(os.path.join(d, f)))
               for f in sorted(os.listdir(d)) if f.endswith(".npy")]

SENTS = [["我", "现在", "要", "去", "学校"],
         ["你", "好", "我", "叫", "名字"],
         ["我", "要", "吃饭"],
         ["谢谢", "你"],
         ["再见"],
         ["现在", "去", "学校", "吃饭"],
         ["名字", "学校", "帮助"],
         ["我", "要", "吃饭", "谢谢", "再见"],
         ["你好", "我", "叫", "名字"],
         ["学校", "什么", "现在"],
         ["我", "要", "帮助", "你"],       # 目标句: 帮助 重录后必测
         ["帮助", "学校", "名字"],          # 三个双手词连打(历史互吃组合)
         ["谢谢", "帮助", "再见"]]


def rs(s, n):
    idx = np.linspace(0, len(s) - 1, n).astype(int)
    return [s[i] for i in idx]


def run(seq, nper, gap, idx):
    """按 nper 帧/词 拼一段连续动作, 喂给 LiveRecognizer, 返回(出的词, 每次出词在第几帧)"""
    frames, ends = [], []
    for w in seq:
        frames += rs(POOL[w][idx % len(POOL[w])], nper)
        ends.append(len(frames) - 1)
        if gap:
            frames += [None] * gap
    rec = Live()
    out, at = [], []
    for i, f in enumerate(frames):
        r = rec.feed(f)
        if r:
            out.append(r[0])
            at.append(i)
    return out, at, ends


def lcs(a, b):
    """最长公共子序列: 容许"多出/漏掉"词, 只统计认对且顺序对的词
       (直接按位置比对的话, 中间多一个词会让后面全部错位, 分虚低)"""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = (dp[i][j] + 1 if a[i] == b[j]
                                else max(dp[i][j + 1], dp[i + 1][j]))
    return dp[-1][-1]


def polish(ws):
    """模拟上屏那一层: 词组合并纠错(和 1.py 里 polish_words 同一个函数)"""
    return G["polish_words"]([[w, 0.4] for w in ws])


print("实时出词验收 (词间无停顿 / 停顿5帧 两种节奏, 已过词组合并纠错)")
print("每词帧数 | 整句全对 | 词级(LCS) | 词级(合并后)")
for nper in (16, 20, 25, 30, 38, 45, 55, 65):
    so = sn = hit = tot = hit2 = 0
    lats = []
    for gap in (0, 5):
        for si, seq in enumerate(SENTS):
            got, at, ends = run(seq, nper, gap, 5 + si)
            pol = polish(got)
            tot += len(seq)
            hit += lcs(seq, got)
            hit2 += lcs(seq, pol)
            so += (pol == seq)
            sn += 1
            if got == seq:
                lats += [a - e for a, e in zip(at, ends)]
    lat = f"  出词延迟{np.mean(lats):.0f}帧" if lats else ""
    print(f"  {nper:>2}帧/词 |  {so:>2}/{sn:<2} {so/sn*100:>3.0f}% |  {hit/tot*100:>5.1f}% |"
          f"  {hit2/tot*100:>5.1f}%{lat}")

print("\n逐句明细 (30帧/词 = 约1秒一个动作, 无停顿; 出的那一列已过词组合并纠错)")
for si, seq in enumerate(SENTS):
    got, at, ends = run(seq, 30, 0, 5 + si)
    pol = polish(got)
    ok = "✓" if pol == seq else "✗"
    print(f"  {ok} 真值 {' '.join(seq):<24} 出的 {' '.join(pol):<24}")

print("\n参数快照:", ", ".join(
    f"{k}={G[k]}" for k in ("JUDGE_WINDOW", "STABLE_VOTES", "SWITCH_VOTES",
                            "SCORE_GATE", "RELEASE_NO_WORD", "JUDGE_EVERY")
    if k in G))
