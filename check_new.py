# -*- coding: utf-8 -*-
# ================= 新录样本体检 =================
# 用法（录完某个词后，先别急着重训，跑一下这个）:
#     python check_new.py 学校
#     python check_new.py            <- 不带参数就检查最近改动的那个词
#
# 它只做一件事: 把你刚录的样本，拿去和"模型里其他所有词"比距离，
# 看会不会被别的词抢走。不用重训、不用开摄像头，几秒钟出结果。
#
# 判定标准（为什么是这个标准）:
#   识别时是"最近 3 个邻居投票 + 相对分数决胜"。所以新样本只要
#   1) 自身散度小(录得稳)  2) 离最像的那个异类词足够远
#   就不会被抢。经验阈值: 到异类的最近距离 >= 自身典型距离的 2 倍。
# ===============================================

import os
import sys
import numpy as np

import train_model as tm

DATA_DIR = tm.DATA_DIR
MODEL_PATH = tm.MODEL_PATH


def load_word(word):
    """读某个词目录下所有原始 .npy，走和训练完全一样的特征管线"""
    wdir = os.path.join(DATA_DIR, word)
    if not os.path.isdir(wdir):
        return None
    out = []
    for f in sorted(os.listdir(wdir)):
        if not f.endswith(".npy"):
            continue
        arr = np.load(os.path.join(wdir, f))
        arr = tm.refine(tm.pad_frame(arr))
        arr[:, 2::3] = 0
        out.append(tm.finalize(arr))
    return np.array(out, dtype=np.float32) if out else None


def motion_of(word):
    """平均帧间变化量: 0.02 以下=基本静止, 0.1 以上=真在做动作"""
    wdir = os.path.join(DATA_DIR, word)
    vals = []
    for f in sorted(os.listdir(wdir)):
        if not f.endswith(".npy"):
            continue
        a = tm.pad_frame(np.load(os.path.join(wdir, f)))
        if len(a) < 3:
            continue
        a = a[:, :126] if a.shape[1] >= 126 else a
        vals.append(float(np.abs(np.diff(a, axis=0)).mean()))
    return float(np.mean(vals)) if vals else 0.0


def relpos_of(word):
    """两手相对位置的中位数: (x, y) 都不为 0 才说明两只手位置错开了"""
    wdir = os.path.join(DATA_DIR, word)
    vals = []
    for f in sorted(os.listdir(wdir)):
        if not f.endswith(".npy"):
            continue
        a = tm.refine(tm.pad_frame(np.load(os.path.join(wdir, f))))
        seq = tm.downsample(a).reshape(tm.FIXED_SEQ, -1)
        vals.append(np.median(seq[:, 63:66], axis=0))
    return np.median(vals, axis=0) if vals else np.zeros(3)


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # 不带参数: 选最近有文件改动的那个词
        best, best_t = None, 0
        for w in sorted(os.listdir(DATA_DIR)):
            p = os.path.join(DATA_DIR, w)
            if not os.path.isdir(p):
                continue
            for f in os.listdir(p):
                if f.endswith(".npy"):
                    t = os.path.getmtime(os.path.join(p, f))
                    if t > best_t:
                        best, best_t = w, t
        target = best
        if not target:
            print("sign_data 里没有样本")
            return
        print(f"(没指定词, 自动检查最近改动的: {target})\n")

    V = load_word(target)
    if V is None or len(V) == 0:
        print(f"[{target}] 没有样本")
        return

    model = np.load(MODEL_PATH, allow_pickle=True)
    MX, My = model["X"].astype(np.float32), model["y"]
    YS = np.array([str(v) for v in My.tolist()])
    words = [str(w) for w in model["thresh_words"].tolist()]
    TH = dict(zip(words, model["thresh_vals"].tolist()))

    # 自身散度: 每条样本到"同词最近另一条"的距离
    D = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    intra = D.min(axis=1)
    typ = float(np.median(intra))

    print(f"=== {target} 新样本体检 ===")
    print(f"样本数 {len(V)}   运动量 {motion_of(target):.3f}   "
          f"自身散度 中位 {typ:.1f} / 最大 {intra.max():.1f}")
    rp = relpos_of(target) / 0.4
    print(f"两手相对位置(中位) x={rp[0]:+.2f}  y={rp[1]:+.2f}  "
          f"<- y 越偏离 0, 越不容易和别的手势混")

    if typ > 6.0:
        print("  ⚠ 自身散度偏大: 每一遍动作差别太大, 门槛会被推高, 容易乱认别的词")

    print()
    print("到其他词的距离 (每条新样本找该词最近的一条, 取中位/最小):")
    rows = []
    for w in words:
        if w == target:
            continue
        idx = np.where(YS == w)[0]
        if len(idx) == 0:
            continue
        dw = np.linalg.norm(V[:, None, :] - MX[idx][None, :, :], axis=2)
        rows.append((float(np.median(dw.min(axis=1))), float(dw.min()), w))

    rows.sort()
    for med, mn, w in rows[:5]:
        flag = ""
        if mn < TH.get(w, 0) * 0.8:
            flag = f"  ← 撞车! {w}门槛{TH.get(w,0):.1f}, 会被它抢"
        elif mn < TH.get(w, 0):
            flag = f"  ← 偏近, {w}门槛{TH.get(w,0):.1f}, 动作一抖就翻"
        print(f"  {w:<5} 中位 {med:6.1f}   最小 {mn:6.1f}{flag}")

    if not rows:
        print("  (模型里没有别的词可比较)")
        return

    med, mn, worst = rows[0]
    th_w = TH.get(worst, 0)
    print()
    print(f"最像的是「{worst}」: 最近距离 {mn:.1f}, 而它的门槛是 {th_w:.1f}")
    # 判定只看一件事: 你这个词有没有落在对方的"接受圈"里。
    # 落在里面 = 对方也能认, 于是靠投票/决胜去抢, 就一定不稳。
    if mn > th_w * 1.15:
        print(f"判定: 通过 —— 已经跳出「{worst}」的接受圈, 可以重训")
    elif mn > th_w:
        print(f"判定: 勉强 —— 刚出圈, 余量只有 {mn - th_w:.1f}, 建议再补 5 遍更夸张的")
    else:
        print(f"判定: 不合格 —— 还在「{worst}」的接受圈里(差 {th_w - mn:.1f}), 重训也会互抢")
        print(f"      要改的: 两手高低差 和 动作幅度, 手型本身已经不是问题了")


if __name__ == "__main__":
    main()
