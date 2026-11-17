# -*- coding: utf-8 -*-
"""单手场景检验: 模拟 MediaPipe 只检测到一只手(第二只手丢了)。

历史 bug: 双手词「学校」在只看到一只手时, 距离被那只不存在的手顶到 20 上下,
过不了自己的双手门槛, 于是被门槛最宽的「帮助」(25.8) 抢走 —— 屏幕上出现"帮助"。
本脚本把每个词的样本都砍掉第二只手再判一次, 直接看谁会掉进谁的圈里。
用法: python mask_hand_check.py
"""
import os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = [("重录前", "sign_model_bak_帮助重录前.npz"),
          ("重录后", "sign_model.npz")]
K = 3
DIM = 126
FIXED_SEQ = 10
SEG = FIXED_SEQ * DIM


def mask_second_hand(mat):
    v = mat.copy()
    s = v[:, :SEG].reshape(len(v), FIXED_SEQ, DIM)
    d = v[:, SEG:].reshape(len(v), FIXED_SEQ, DIM)
    s[:, :, 63:] = 0
    d[:, :, 63:] = 0
    return v


def check(path):
    m = np.load(os.path.join(BASE, path), allow_pickle=False)
    X = mask_second_hand(m["X"].astype(np.float32))
    y = np.array([str(v) for v in m["y"].tolist()])
    words = [str(w) for w in m["thresh_words"].tolist()]
    TH1 = dict(zip(words, m["thresh1_vals"].tolist()))
    # 和 1.py 一致: 单手状态下双手词直接出局(帮助/学校/名字/现在 全都是双手词)
    TWO = {w: bool(v) for w, v in zip(words, m["two_hand_vals"].tolist())}
    WORD_IDX = {w: np.where(y == w)[0] for w in words}
    killed = [w for w in words if TWO.get(w)]
    print(f"  单手状态下被排除的双手词: {' '.join(killed)}")
    # wrong  = 认错且过了对方的门槛 -> 屏幕上会真的显示错的词(最坏)
    # mute   = 认错但没过对方门槛 -> 只是不出词(可接受)
    res = {w: {"ok": 0, "rej": 0, "wrong": {}, "mute": {}} for w in words}
    for i in range(len(y)):
        d = np.linalg.norm(X - X[i], axis=1)
        d[i] = np.inf                      # 留一: 不许自己给自己当邻居
        for w in killed:                   # 双手词出局
            d[WORD_IDX[w]] = np.inf
        yy = y
        order = np.argsort(d)[:K]
        votes = [yy[j] for j in order]
        pred = max(set(votes), key=votes.count)
        w = y[i]
        dd = d[order[0]]
        if pred == w:
            if dd < TH1.get(w, 4.0):
                res[w]["ok"] += 1
            else:
                res[w]["rej"] += 1          # 认对但过不了自己的门槛 -> 不出词
        elif dd < TH1.get(pred, 4.0):
            res[w]["wrong"][pred] = res[w]["wrong"].get(pred, 0) + 1
        else:
            res[w]["mute"][pred] = res[w]["mute"].get(pred, 0) + 1
    return words, res, TH1, killed


for tag, path in MODELS:
    if not os.path.exists(os.path.join(BASE, path)):
        print(f"[跳过] {path}")
        continue
    words, res, TH1, killed = check(path)
    # 双手词在单手状态下按设计不出词, 不参与"单手词准不准"的统计
    ones = [w for w in words if w not in killed]
    tot = sum(res[w]["ok"] + res[w]["rej"] + sum(res[w]["wrong"].values())
              + sum(res[w]["mute"].values()) for w in ones)
    ok = sum(res[w]["ok"] for w in ones)
    bad_all = sum(sum(res[w]["wrong"].values()) for w in ones)
    print(f"\n===== 单手场景 {tag} ({path}) =====")
    print(f"  单手词认对 {ok}/{tot} = {ok/tot*100:.0f}%   会显示错词 {bad_all} 条")
    print(f"  双手词(名字/学校/帮助/现在)在这一幕里按设计不出词 —— 不计入上面")
    print(f"  {'词':<6}{'认对':>6}{'不出词':>7}{'门槛1':>7}   会显示错词 | 认错但被门槛挡掉")
    for w in sorted(ones, key=lambda z: -(res[z]["rej"] + sum(res[z]["wrong"].values()))):
        r = res[w]
        bad = res[w]["rej"] + sum(r["wrong"].values()) + sum(r["mute"].values())
        if bad == 0:
            continue
        wr = " ".join(f"{k}x{v}" for k, v in sorted(r["wrong"].items(), key=lambda t: -t[1]))
        mu = " ".join(f"{k}x{v}" for k, v in sorted(r["mute"].items(), key=lambda t: -t[1]))
        print(f"  {w:<6}{r['ok']:>6}{r['rej']:>7}{TH1[w]:>7.1f}   {wr or '-':<14}| {mu or '-'}")
    print("  (只列出有问题的词; 没列出的在单手状态下全部认对)")
