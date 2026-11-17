# -*- coding: utf-8 -*-
# ================= 麦克风链路验收 =================
# 用法: 接好线、ESP32 通电后运行
#     "D:/python/python.exe" voice_diag.py
# 然后对着麦克风说几句话。跑完自动给出判定:
#   宽带正常   -> 95%能量截止频率 > 2500Hz, 修好了
#   仍然窄带   -> 还是 ~400Hz 低通, 线还是坏的
# =================================================

import time
import wave
import numpy as np
import paho.mqtt.client as mqtt

BROKER = "broker.hivemq.com"
TOPIC = "aiot/voice/audio"
DURATION = 180          # 最长等 3 分钟
MIN_SEGMENTS = 3        # 收到 3 段就开始分析

segs = []
done = False

def on_connect(c, u, f, rc, props=None):
    print(f"[MQTT] 已连接 (rc={rc}), 监听 {TOPIC} ...")
    c.subscribe(TOPIC, qos=0)

def on_message(c, u, msg):
    if len(msg.payload) < 3200:          # < 0.2 秒的忽略
        return
    pcm = np.frombuffer(msg.payload[:len(msg.payload)//2*2], dtype=np.int16)
    segs.append(pcm)
    print(f"  [收到] 第 {len(segs)} 段, {len(pcm)/8000:.1f} 秒, "
          f"峰值 {int(np.abs(pcm.astype(np.float32)).max())}")
    if len(segs) >= MIN_SEGMENTS:
        globals()["done"] = True

def analyze():
    print()
    print("=== 频谱体检 ===")
    bad = 0
    for k, x in enumerate(segs):
        xf = x.astype(np.float64) - x.mean()
        if np.abs(xf).max() < 500:
            print(f"  第{k+1}段: 几乎无声 (峰值 {int(np.abs(xf).max())}), 跳过")
            continue
        n = 4096
        acc = np.zeros(n // 2 + 1)
        for s in range(0, max(1, len(xf) - n), n // 2):
            acc += np.abs(np.fft.rfft(xf[s:s+n] * np.hanning(n))) ** 2
        freq = np.fft.rfftfreq(n, 1 / 8000)
        cum = np.cumsum(acc) / acc.sum()
        cut95 = freq[min(np.searchsorted(cum, 0.95), len(freq) - 1)]
        hi = acc[freq > 1000].sum() / acc.sum() * 100
        verdict = "宽带 OK" if cut95 > 2500 else ("勉强" if cut95 > 1000 else "窄带!")
        if cut95 <= 1000:
            bad += 1
        print(f"  第{k+1}段: 95%能量截止 {cut95:6.0f} Hz | 1kHz以上占比 {hi:5.1f}%  -> {verdict}")
    print()
    ok = sum(1 for x in segs if np.abs(x.astype(np.float64) - x.mean()).max() > 500)
    if ok == 0:
        print("判定: 没收到有效声音 —— 线可能完全断开, 或没触发录音")
    elif bad == 0:
        print("判定: 宽带正常, 链路已修复! 直接跑 1.py (VOICE_SOURCE 改回 esp32)")
    elif bad < ok:
        print("判定: 部分段正常 —— 线还是接触不良, 重插/换线再测")
    else:
        print("判定: 仍然窄带 —— OUT/GND 线还是坏的, 换新杜邦线再测")
    # 顺手存档, 方便远端分析
    allx = np.concatenate(segs)
    with wave.open("voice_diag_last.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(allx.astype(np.int16).tobytes())
    print("(音频已存到 voice_diag_last.wav, 可反复分析)")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2 if hasattr(mqtt, "CallbackAPIVersion") else None)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883, 60)
client.loop_start()

t0 = time.time()
while time.time() - t0 < DURATION and not done:
    time.sleep(0.5)

if not segs:
    print(f"{DURATION} 秒内一个音频包都没收到 —— ESP32 没在发, 检查它是否在线/开口有没有触发")
else:
    analyze()
client.loop_stop()
