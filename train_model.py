# -*- coding: utf-8 -*-
# ================= 手语模型训练脚本 =================
# 用法: 录完样本后运行  python train_model.py
# 做的事:
#   1. 读取 sign_data/<词名>/ 下所有 .npy 样本
#   2. 每条 45 帧样本降采样成 10 帧, 拉平成向量 (保留动作轨迹)
#      双手版每帧 126 维 (右手63+左手63), 拉平后 1260 维
#      旧的单手样本(63维)自动补零成 126 维, 与新样本混着用, 不用作废
#   3. 训练 kNN 分类器 (留 20% 样本当考卷, 报告准确率)
#   4. 模型保存为 sign_model.npz
# 注意: 识别脚本 sign_recognize.py 加载的就是这个文件
#       每次录了新样本, 重跑一遍本脚本即可, 词库扩大不用改任何代码
# ====================================================

import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sign_data")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sign_model.npz")

FIXED_SEQ = 10   # 每条样本降采样到 10 帧 (与识别脚本保持一致, 改要两边一起改)
DIM = 126        # 每帧特征维度: 双手 63+63

def pad_frame(arr):
    """旧单手样本 (T,63) 补零成 (T,126), 双手样本原样返回"""
    if arr.ndim == 2 and arr.shape[1] < DIM:
        arr = np.hstack([arr, np.zeros((arr.shape[0], DIM - arr.shape[1]), dtype=arr.dtype)])
    return arr

def refine(arr):
    """双手特征降噪: B手以自己手腕为中心重新取形状(稳定部分),
       '两手相对位置'只保留在 B手手腕 3 个维度上并降权 0.4。
       效果: 双手词每遍'两手远近高低'抖动造成的方差大幅下降。
       单手样本 B手全零, 经此函数不变, 完全兼容。"""
    if arr.ndim != 2 or arr.shape[1] != DIM:
        return arr
    b = arr[:, 63:].reshape(arr.shape[0], 21, 3).copy()
    wrist = b[:, 0:1, :]                      # B手手腕 = 两手相对位置
    centered = b - wrist                      # B手形状(以自己手腕为原点)
    centered[:, 0, :] = wrist[:, 0, :] * 0.4  # 相对位置降权后放回手腕位
    return np.concatenate([arr[:, :63], centered.reshape(arr.shape[0], 63)], axis=1)

def downsample(arr, n=FIXED_SEQ):
    """(T, 126) 序列 -> (n, 126) 等间隔取帧 -> 拉平"""
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return arr[idx].flatten()

MOTION_W = 0.7   # 运动特征权重

def finalize(arr):
    """静态轨迹 + 运动特征(帧间变化量): 让'拇指弯两下'这类动态差异
       被显式表达, 区分 好(拇指不动) 和 谢谢(拇指弯) 这类形状相近的词。
       训练和识别必须用同一个函数。"""
    ds = downsample(arr)                          # (840,)
    seq = ds.reshape(FIXED_SEQ, -1)
    delta = np.zeros_like(seq)
    delta[1:] = seq[1:] - seq[:-1]                # 相邻帧变化量
    return np.concatenate([ds, delta.flatten() * MOTION_W])

def load_dataset():
    X, y = [], []
    if not os.path.isdir(DATA_DIR):
        return None, None
    for word in sorted(os.listdir(DATA_DIR)):
        wdir = os.path.join(DATA_DIR, word)
        if not os.path.isdir(wdir):
            continue
        for f in sorted(os.listdir(wdir)):
            if not f.endswith(".npy"):
                continue
            arr = np.load(os.path.join(wdir, f))
            arr = refine(pad_frame(arr))
            arr[:, 2::3] = 0   # 去掉z维度(MediaPipe深度噪声大), 与识别脚本一致
            X.append(finalize(arr))
            y.append(word)
    if not X:
        return None, None
    return np.array(X, dtype=np.float32), np.array(y)

def main():
    X, y = load_dataset()
    if X is None:
        print(f"没有找到样本, 请先运行 sign_collect.py 录制, 数据目录: {DATA_DIR}")
        return

    words, counts = np.unique(y, return_counts=True)
    print("=== 样本统计 ===")
    for w, c in zip(words, counts):
        flag = "OK " if c >= 20 else "不足"
        print(f"  {w}: {c} 条 [{flag}]")
    print(f"合计 {len(y)} 条\n")

    # ---- 划分 训练集/考卷集 (每个词 80% 训练, 20% 考试) ----
    rng = np.random.default_rng(42)
    train_idx, test_idx = [], []
    for w in words:
        idx = np.where(y == w)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * 0.2))
        test_idx += list(idx[:n_test])
        train_idx += list(idx[n_test:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)

    # ---- kNN 分类: 考卷每条在训练集里找最近的 k 个邻居投票 ----
    K = 3
    correct = 0
    confusions = {}
    for i in test_idx:
        d = np.linalg.norm(X[train_idx] - X[i], axis=1)
        top = np.argsort(d)[:K]
        votes = [y[train_idx][j] for j in top]
        pred = max(set(votes), key=votes.count)
        if pred == y[i]:
            correct += 1
        else:
            key = f"{y[i]} -> 误识为 {pred}"
            confusions[key] = confusions.get(key, 0) + 1

    acc = correct / len(test_idx) * 100 if len(test_idx) else 0
    print(f"=== 考试结果: {correct}/{len(test_idx)}  准确率 {acc:.1f}% ===")
    if confusions:
        print("容易混淆的:")
        for k, v in sorted(confusions.items(), key=lambda t: -t[1]):
            print(f"  {k}  x{v}")

    # ---- 每个词的自适应门槛 ----
    # 关键改动 1: 门槛下限按"单手/双手"分开。
    #   单手词: 动作简单、特征稳, 下限 8.0
    #   双手词: 两只手各自 42 维自由度, 两手相对位置每遍都在变,
    #           实时比划的实际距离天然 15~25, 门槛给个位数等于永远打不中。
    #           下限 20.0
    # 关键改动 2: 再算一套"单手门槛"。
    #   实测: 双手词样本把第二只手清零后, 跟自家样本的距离是 17~20 ——
    #   说明实时"只检测到一只手"时, 距离会被这只不存在的手顶到 20,
    #   于是过不了双手门槛, 被别的词抢走(学校 -> 帮助 就是这么来的)。
    #   所以识别端在单手状态下改用"单手版样本 + 单手门槛"。
    SEG = FIXED_SEQ * DIM

    def mask_second_hand(mat):
        v = mat.copy()
        s = v[:, :SEG].reshape(len(v), FIXED_SEQ, DIM)
        d = v[:, SEG:].reshape(len(v), FIXED_SEQ, DIM)
        s[:, :, 63:] = 0
        d[:, :, 63:] = 0
        return v

    def hand_ratio(i):
        """这条样本里'两手都在画面'的帧占比"""
        seq = X[i][:SEG].reshape(FIXED_SEQ, DIM)
        return float((np.abs(seq[:, 63:]).sum(axis=1) > 1e-6).mean())

    TWO_HAND = {w: float(np.mean([hand_ratio(i) for i in np.where(y == w)[0]])) > 0.5
                for w in words}

    def build_thresh(M, floor_two, floor_one):
        th, info, typ = {}, {}, {}
        for w in words:
            idx = np.where(y == w)[0]
            # ---- 只用"录得完整"的样本估门槛 ----
            # 实测: 帮助 20 条里 12 条有 10% 以上的帧只检测到一只手,
            # 这些"半残"样本把自身散度撑到 11.3, 门槛跟着涨到 26.8,
            # 于是它的接受圈把"学校"整个盖住(学校离它 18.6) —— 学校被抢走。
            # 残缺样本仍然留在样本库里参与比对(不能删, 删了就少了覆盖),
            # 但不再参与"门槛该多松"的计算。
            if TWO_HAND[w]:
                hr = [hand_ratio(i) for i in idx]
                clean = idx[np.array(hr) >= 0.9]
                if len(clean) < 6:          # 完整的太少就退回全部, 免得估偏
                    clean = idx
            else:
                clean = idx
            intra = []
            for i in clean:
                d = np.linalg.norm(M[clean] - M[i], axis=1)
                d = d[d > 1e-6]
                if len(d):
                    intra.append(d.min())
            p97 = float(np.percentile(intra, 97)) if intra else 5.0
            # 典型距离(中位数): 这个词"每遍之间本来差多少"。
            # 名字 4.6、学校 2.6 —— 尺度差一倍, 用绝对距离比时名字天然吃亏,
            # 识别端要拿它当分母做归一化, 才不会"谁散谁输"。
            typ[w] = float(np.median(intra)) if intra else 4.0
            floor = floor_two if TWO_HAND[w] else floor_one
            others = np.where(y != w)[0]
            if len(others):
                dx = np.linalg.norm(M[idx][:, None, :] - M[others][None, :, :], axis=2)
                cross = float(np.median(dx.min(axis=1)))
                # 到别的词最近能有多近 —— 门槛必须低于它, 否则两个词的圈会重叠
                cross_min = float(dx.min())
            else:
                cross, cross_min = 999.0, 999.0
            base = max(p97 * 1.7, floor)
            # 注意: 这里不再往上"封顶"。
            # 以前想用"到最近异类词的距离"限制门槛, 结果反而造成
            # 名字(4.0) / 学校(2.5) 这种尺度不同的词互相吃。
            # 现在分工明确: 门槛只决定"要不要接受", 选哪个词由识别端的
            # 相对分数(d / 典型距离)决定 —— 见 1.py。
            th[w] = base
            info[w] = (p97, cross, base, th[w], typ[w], cross_min)
        return th, info, typ

    th, thinfo, TY = build_thresh(X, 20.0, 8.0)          # 双手状态用的门槛
    th1, _, TY1 = build_thresh(mask_second_hand(X), 8.0, 8.0)   # 单手状态用的门槛

    print("=== 每词门槛 (双手下限20 / 单手下限8, 但不超过混淆边界) ===")
    print(f"  {'词':<6}{'手':>4}{'自身p97':>9}{'最近异类':>9}{'典型':>7}{'双手门槛':>10}{'单手门槛':>10}")
    for w in words:
        p97, cross, base, v, t, cmin = thinfo[w]
        tag = "双" if TWO_HAND[w] else "单"
        note = ""
        if abs(v - base) > 1e-6:
            note = f" ←压到{cmin:.1f}的0.9倍"
        print(f"  {w:<6}{tag:>4}{p97:>9.1f}{cross:>9.1f}{t:>7.1f}{v:>10.1f}{th1[w]:>10.1f}{note}")

    # ---- 每词"运动指纹": 该词教材里每个维度的平均运动量 (84维) ----
    # 识别端决胜用: 两个形状相近的词票数接近时, 挑差异最大的维度,
    # 比较直播窗口在这些维度上的运动量, 谁像选谁 (通用机制, 不针对词)
    def word_delta_profile(words_):
        prof = {}
        for w in words_:
            idx = np.where(y == w)[0]
            es = []
            for i in idx:
                seq = X[i][:840].reshape(FIXED_SEQ, -1)
                es.append(np.abs(np.diff(seq, axis=0)).mean(axis=0))
            prof[w] = np.mean(es, axis=0)
        return prof

    prof = word_delta_profile(words)
    print("=== 每词运动指纹已生成 ===")

    # ---- 保存模型: 样本 + 标签 + 门槛 + 运动指纹 ----
    np.savez_compressed(MODEL_PATH, X=X, y=y, fixed_seq=FIXED_SEQ,
                        thresh_words=np.array(words),
                        thresh_vals=np.array([th[w] for w in words], dtype=np.float32),
                        thresh1_vals=np.array([th1[w] for w in words], dtype=np.float32),
                        typical_vals=np.array([TY[w] for w in words], dtype=np.float32),
                        typical1_vals=np.array([TY1[w] for w in words], dtype=np.float32),
                        two_hand_vals=np.array([1 if TWO_HAND[w] else 0 for w in words],
                                               dtype=np.int8),
                        delta_words=np.array(words),
                        delta_matrix=np.array([prof[w] for w in words], dtype=np.float32))
    print(f"\n模型已保存: {MODEL_PATH}")
    print("下一步: 运行 sign_recognize.py 实时识别")

if __name__ == "__main__":
    main()
