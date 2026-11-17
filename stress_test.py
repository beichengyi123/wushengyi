# -*- coding: utf-8 -*-
"""压力测试: 专测真实使用中会遇到的"脏"情况, 改造前 vs 改造后 A/B。
   compare_ab.py 只喂干净的连续比划, 真实使用远不止这些:
   手抬着不动 / 半吊子的过渡动作 / 同一个词连打两次 / 摄像头掉帧 ...
   这些场景最能暴露"该出词没出 / 不该出词乱出"。
"""
from ab_lib import NEW, OLD, lcs, run, polish, arr_to_frames, rs, POOL, SENTS


def word_frames(w, n, idx):
    return rs(arr_to_frames(POOL[w][idx % len(POOL[w])]), n)


def chimera(w1, w2, n, idx):
    """半个词A + 半个词B 拼起来 = 一个"半吊子"姿势。
       手在换动作的半途就是这个样子, 不该被认成任何一个词。"""
    return word_frames(w1, n, idx)[:n // 2] + word_frames(w2, n, idx + 3)[n // 2:]


def s_plain(seq, si):                  # ① 基准: 正常连贯比划
    fr = []
    for w in seq:
        fr += word_frames(w, 35, 5 + si)
    return fr


def s_head_tail(seq, si):              # ② 句首句尾手抬着瞎动(还没开始/已经结束)
    fr = chimera("我", "谢谢", 25, si)
    for w in seq:
        fr += word_frames(w, 35, 5 + si)
    fr += chimera("再见", "学校", 25, si)
    return fr


def s_transition(seq, si):             # ③ 每个词之间夹一段"半吊子"过渡
    fr = []
    for k, w in enumerate(seq):
        if k:
            fr += chimera(seq[k - 1], w, 15, si)
        fr += word_frames(w, 35, 5 + si)
    return fr


def s_hold(seq, si):                   # ④ 比完一个词手停着不动(等对方看完)
    fr = []
    for w in seq:
        fr += word_frames(w, 35, 5 + si)
        fr += rs(arr_to_frames(POOL[w][(5 + si) % len(POOL[w])]), 60)
        fr += [None] * 6
    return fr


def s_repeat(seq, si):                 # ⑤ 同一个词连打两次(中间手放下)
    return (word_frames(seq[0], 35, 5 + si) + [None] * 12 +
            word_frames(seq[0], 35, 5 + si) + [None] * 12)


def s_lowfps(seq, si):                 # ⑥ 摄像头只有 15fps: 每两帧丢一帧
    fr = []
    for w in seq:
        blk = word_frames(w, 35, 5 + si)
        fr += [blk[i] for i in range(0, len(blk), 2)]
    return fr


# (场景名, 构造函数, 期望输出 = None表示就是原句, 场景fps)
SCENES = [
    ("正常连贯比划", s_plain, None, 30.0),
    ("句首句尾手瞎动", s_head_tail, None, 30.0),
    ("词间半吊子过渡", s_transition, None, 30.0),
    ("比完手停着不动", s_hold, None, 30.0),
    ("同词连打两次", s_repeat, "twice", 30.0),
    ("摄像头仅15fps", s_lowfps, None, 15.0),
]

print("压力测试: 每个场景 5 句, 改造前 vs 改造后")
print("  (词级=认对且顺序对的百分比; 多词=凭空多认出来; 漏词=该认没认到)\n")
print("场景              版本      整句全对     词级      多词   漏词")
for name, gen, kind, fps in SCENES:
    NEW["apply_fps"](fps)          # 新版按帧率自适应; 旧版没这个函数, 保持不变
    for label, G in (("改造前", OLD), ("改造后", NEW)):
        so = sn = hit = tot = extra = miss = 0
        for si, seq in enumerate(SENTS):
            want = [seq[0], seq[0]] if kind == "twice" else seq
            refp = polish(NEW, want)
            got = polish(G, run(G, gen(seq, si)))
            h = lcs(refp, got)
            hit += h
            tot += len(refp)
            extra += len(got) - h
            miss += len(refp) - h
            so += (got == refp)
            sn += 1
        print("  %-14s %-6s   %2d/%2d %3.0f%%   %5.1f%%    %3d    %3d"
              % (name, label, so, sn, so / sn * 100, hit / tot * 100, extra, miss))
    print()
