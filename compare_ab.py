# -*- coding: utf-8 -*-
"""A/B 对比: 改造前 vs 改造后 (基线由 ab_lib.downgrade 重建, 不依赖备份文件)
   除了干净的连续比划, 还模拟真实使用里常见的两种脏数据:
     MediaPipe 偶尔漏检一只手 / 手部关键点抖动
"""
import numpy as np
from ab_lib import NEW, OLD, lcs, run, polish, make_frames, SENTS


def evaluate(tag, p_miss, sigma):
    """真值也会过一遍 polish_words: 否则真值['你','好',...] 对输出['你好',...]
       会被误判成漏词 —— 而"单字合并成词"本来就是这次要实现的功能, 不能算错。"""
    print(f"\n=== {tag}  (漏检概率{p_miss:.0%}, 抖动σ={sigma}) ===")
    print("   版本     整句全对   词级(LCS)   多出的词   漏掉的词")
    for name, G in (("改造前", OLD), ("改造后", NEW)):
        rng = np.random.default_rng(42)
        so = sn = hit = tot = extra = miss = 0
        for nper in (30, 45):
            for gap in (0, 5):
                for si, seq in enumerate(SENTS):
                    seq_ref = polish(NEW, seq)      # 两边同一套合并规则
                    got = polish(G, run(G, make_frames(seq, nper, gap, 5 + si, rng,
                                                       p_miss, sigma)))
                    h = lcs(seq_ref, got)
                    hit += h; tot += len(seq_ref)
                    extra += len(got) - h; miss += len(seq_ref) - h
                    so += (got == seq_ref); sn += 1
        print("   %-8s  %2d/%2d %3.0f%%    %5.1f%%     %4d       %4d"
              % (name, so, sn, so / sn * 100, hit / tot * 100, extra, miss))
        if name == "改造前":
            base = (so / sn * 100, hit / tot * 100)
        else:
            print("   提升      %+5.0f个百分点整句  %+5.1f个百分点词级"
                  % (so / sn * 100 - base[0], hit / tot * 100 - base[1]))


print("A/B 对比: 输入数据完全相同(同一随机种子), 只换 LiveRecognizer 这一层")
evaluate("干净流: 正常连贯比划", 0.0, 0.0)
evaluate("轻度脏: 偶尔漏检一帧手", 0.05, 0.0)
evaluate("较脏: 漏检 + 手部抖动(真实摄像头的样子)", 0.10, 0.012)
