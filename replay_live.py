# -*- coding: utf-8 -*-
# 真机复盘: 读 sign_live/rec_*.npy (按 Enter 时自动存的原始动作帧),
# 把每块、每种窗口长度分别识别成什么、分数多少全打出来。
# 用法:  python replay_live.py            <- 自动用最新一份
#        python replay_live.py 某文件.npy
import os, sys, glob, time
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
     "MIN_SEG": 10, "MAX_SEG": 75, "SEG_SILENCE": 4,
     "SCAN_WINS": (24, 32, 45, 60)}
exec(compile(b1, "1.py", "exec"), G)
exec(compile(b2, "1.py", "exec"), G)
G["WORD_IDX"] = {w: np.where(G["MY_S"] == w)[0] for w in np.unique(G["MY_S"])}
TH, classify, blocks = G["TH"], G["_classify_frames"], G["_blocks"]

path = sys.argv[1] if len(sys.argv) > 1 else None
if path is None:
    cand = sorted(glob.glob(os.path.join(BASE, "sign_live", "rec_*.npy")))
    if not cand:
        print("sign_live/ 里还没有存档 —— 先在 1.py 里比划一句再按 Enter")
        sys.exit()
    path = cand[-1]
print(f"读入: {os.path.basename(path)}")
frames = list(np.load(path, allow_pickle=True))
print(f"共 {len(frames)} 帧, 其中没手的 {sum(1 for f in frames if f is None)} 帧 "
      f"(约 {len(frames)/30:.1f} 秒)")

bl = blocks(frames)
print(f"\n切成 {len(bl)} 块: " + ", ".join(f"{len(b)}帧" for b in bl))

print("\n=== 每块: 整块直接分类 + 各种窗长的识别结果 ===")
for i, b in enumerate(bl):
    r = classify(b)
    if r:
        print(f"\n[块{i+1}] {len(b)}帧  整块 -> {r[0]}  距离{r[1]:.1f}/门槛{TH.get(r[0],0):.1f} "
              f"{'过' if r[1] < TH.get(r[0], 0) else '超门槛'}")
    else:
        print(f"\n[块{i+1}] {len(b)}帧  整块 -> 没结果")
    for W in G["SCAN_WINS"]:
        if len(b) < W * 0.7:
            continue
        p = G["_picks"](b, W, step=3)
        if not p:
            print(f"   窗长{W:>2}: (没有过门槛的位置)")
            continue
        txt = "  ".join(f"+{c[1]}帧:{c[2]}({c[0]:.2f})" for c in p)
        print(f"   窗长{W:>2}: {txt}")

print("\n=== 最终结果 (1.py 当前参数) ===")
words, details, cost = G["batch_recognize"](frames)
print(f"识别 {len(words)} 词: {' + '.join(words) if words else '(空)'}   计算{cost:.2f}s")
try:
    import sentence_api
    print(f"组句: {sentence_api.compose(words)}")
except Exception as e:
    print(f"组句失败: {e}")
print("\n(括号里是分数 = 距离/门槛, 越小越像; 大于 1 就是过不了门槛)")
