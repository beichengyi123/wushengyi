# -*- coding: utf-8 -*-
# 用 1.py 里的真实函数做验收: 多种句子 + 不同停顿节奏 + 换样本
import os, time
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(BASE, "1.py"), encoding="utf-8").read()
b1 = SRC[SRC.index("def _normalize_seq("):SRC.index("# ================= 功能 1.5")]
b2 = SRC[SRC.index("# ================= 功能 1a"):SRC.index("# ================= 功能 1：手语识别")]

m = np.load(os.path.join(BASE, "sign_model.npz"), allow_pickle=False)
X = m["X"].astype(np.float32); My = m["y"]
words_all = [str(w) for w in m["thresh_words"].tolist()]
G = {"np": np, "time": time, "threading": __import__("threading"),
     "MX": X, "My": My,
     "TH": dict(zip(words_all, m["thresh_vals"].tolist())),
     "TY": dict(zip(words_all, m["typical_vals"].tolist())),
     "TWO": {w: bool(v) for w, v in zip(words_all, m["two_hand_vals"].tolist())},
     "MY_S": np.array([str(v) for v in My.tolist()]),
     "K": 3, "FIXED_SEQ": int(m["fixed_seq"]), "MOTION_W": 0.7,
     "DIST_THRESHOLD": 4.0, "RECORD_FRAMES": 45, "SEG_DIM": int(m["fixed_seq"]) * 126,
     "MIN_SEG": 10, "MAX_SEG": 75, "SEG_SILENCE": 4}
exec(compile(b1, "1.py", "exec"), G)
exec(compile(b2, "1.py", "exec"), G)
G["WORD_IDX"] = {w: np.where(G["MY_S"] == w)[0] for w in np.unique(G["MY_S"])}
batch = G["batch_recognize"]


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
         ["我", "要", "吃饭"], ["谢谢", "你"], ["再见"],
         ["现在", "去", "学校", "吃饭"],
         ["名字", "学校", "帮助"],
         ["我", "要", "吃饭", "谢谢", "再见"],
         ["你好", "我", "叫", "名字"],
         ["学校", "什么", "现在"]]

rows, tot_w, hit_w, sent_ok, sent_n = [], 0, 0, 0, 0
tcosts = []
for gap in (0, 3, 5, 10):
    for si, seq in enumerate(SENTS):
        idx = 5 + si
        frames = []
        for w in seq:
            frames += POOL[w][idx % len(POOL[w])]
            if gap:
                frames += [None] * gap
        t0 = time.time()
        got, det, cost = batch(frames)
        tcosts.append(cost)
        h = sum(1 for a, b in zip(seq, got) if a == b)
        exact = (got == seq)
        tot_w += len(seq); hit_w += h; sent_n += 1; sent_ok += exact
        rows.append((gap, " ".join(seq), " ".join(got), f"{h}/{len(seq)}", exact, cost))

print(f"{'停顿':<5}{'真值':<26}{'识别':<26}{'词':>6}  整句  {'耗时':>6}")
print("-" * 84)
for gap, t, g, h, e, c in rows:
    print(f"{gap:<6}{t:<26}{g:<26}{h:>6}  {'✓' if e else '✗':<4} {c:>6.2f}s")
print("-" * 84)
print(f"整句完全正确: {sent_ok}/{sent_n} = {sent_ok/sent_n*100:.0f}%")
print(f"词级正确率  : {hit_w}/{tot_w} = {hit_w/tot_w*100:.1f}%")
print(f"识别耗时    : 平均 {np.mean(tcosts):.2f}s  最长 {max(tcosts):.2f}s")
