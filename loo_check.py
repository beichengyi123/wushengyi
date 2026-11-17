# -*- coding: utf-8 -*-
"""留一验证(LOO): 每条样本轮流当考卷, 其余当题库 —— 不再拿训练样本自己回放自己。

比的是「重录前后」两个模型:
  1. 每词准确率 / 拒识率(门槛没过)
  2. 误吸: 别的词的样本被判成了这个词(门槛太宽的代价)
用法: python loo_check.py
"""
import os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = [("重录前", "sign_model_bak_帮助重录前.npz"),
          ("重录后", "sign_model.npz")]
K = 3


def loo(path):
    m = np.load(os.path.join(BASE, path), allow_pickle=False)
    X = m["X"].astype(np.float32)
    y = np.array([str(v) for v in m["y"].tolist()])
    words = [str(w) for w in m["thresh_words"].tolist()]
    TH = dict(zip(words, m["thresh_vals"].tolist()))
    TY = dict(zip(words, m["typical_vals"].tolist()))

    stat = {w: {"ok": 0, "rej": 0, "wrong": 0, "n": 0} for w in words}
    absorb = {w: [] for w in words}     # 被判成 w 但真值不是 w 的词
    confuse = {}
    for i in range(len(y)):
        d = np.linalg.norm(np.delete(X, i, axis=0) - X[i], axis=1)
        yy = np.delete(y, i, axis=0)
        order = np.argsort(d)[:K]
        votes = [yy[j] for j in order]
        pred = max(set(votes), key=votes.count)
        w = y[i]
        stat[w]["n"] += 1
        if pred != w:
            stat[w]["wrong"] += 1
            absorb.setdefault(pred, []).append(w)
            confuse[f"{w}->{pred}"] = confuse.get(f"{w}->{pred}", 0) + 1
        elif d[order[0]] >= TH.get(w, 4.0):
            stat[w]["rej"] += 1        # 认对了但没过门槛 -> 实际用起来是"不出词"
        else:
            stat[w]["ok"] += 1
    return words, stat, absorb, confuse, TH, TY


for tag, path in MODELS:
    if not os.path.exists(os.path.join(BASE, path)):
        print(f"[跳过] 找不到 {path}")
        continue
    words, stat, absorb, confuse, TH, TY = loo(path)
    tot = sum(s["n"] for s in stat.values())
    ok = sum(s["ok"] for s in stat.values())
    rej = sum(s["rej"] for s in stat.values())
    print(f"\n===== {tag} ({path}) =====")
    print(f"  整体: 认对+过门槛 {ok}/{tot} = {ok/tot*100:.1f}%   "
          f"认对但被门槛挡掉(不出词) {rej}   认错 {tot-ok-rej}")
    print(f"  {'词':<6}{'条数':>5}{'可用%':>8}{'不出词':>7}{'认错':>6}{'门槛':>7}   误吸(别人被判成它)")
    for w in sorted(words, key=lambda z: -stat[z]["rej"] - stat[z]["wrong"]):
        s = stat[w]
        if s["n"] == 0:
            continue
        ab = {}
        for x in absorb.get(w, []):
            ab[x] = ab.get(x, 0) + 1
        ab_s = " ".join(f"{k}x{v}" for k, v in sorted(ab.items(), key=lambda t: -t[1]))
        print(f"  {w:<6}{s['n']:>5}{s['ok']/s['n']*100:>8.0f}{s['rej']:>7}{s['wrong']:>6}"
              f"{TH[w]:>7.1f}   {ab_s}")
    if confuse:
        print("  混淆明细: " + ", ".join(f"{k}({v})" for k, v in
                                         sorted(confuse.items(), key=lambda t: -t[1])[:8]))
