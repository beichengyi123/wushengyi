# -*- coding: utf-8 -*-
# ================= 电脑麦克风 -> 百度识别 =================
# 作用和 ESP32 那条语音通道一样, 但音源换成电脑自己的麦克风(耳机麦/耳麦/USB麦)。
# 好处: 电脑麦克风是 16kHz 宽带, 高频完整, 识别率远比 ESP32 那根模拟线高。
# 用法(在 1.py 里):  VOICE_SOURCE = "pc"
#
# 单独运行本文件可以当"语音输入测试器": 对着麦克风说话, 看能不能识别出字。
# ==========================================================

import queue
import threading
import time

import numpy as np

try:
    import sounddevice as sd
except Exception as e:          # 没装库时给出明确提示, 不让主程序崩
    sd = None
    _SD_ERROR = e

import voice_api

RATE = 16000            # 电脑麦克风用 16kHz, 百度原生支持
BLOCK_SEC = 0.1         # 每 0.1 秒判断一次音量
PRE_ROLL = 4            # 开口前先留 0.4 秒, 免得第一个字被切掉
QUIET_BLOCKS = 6        # 连续 0.6 秒安静 = 这句说完了
MAX_SEC = 6.0           # 单句最长 6 秒, 防止一直录
TRIGGER_BLOCKS = 2      # 连续 2 块(0.2 秒)都超门槛才算开口, 挡掉咳嗽敲桌子
ABS_MIN = 150.0         # 门槛下限(int16 幅度), 房间再安静也不会低于这个


class PcMic:
    """电脑麦克风 + 自动断句 + 百度识别, 识别结果通过 on_text 回调交出去"""

    def __init__(self, on_text, device=None):
        self.on_text = on_text
        self.device = device
        self.blocks = queue.Queue()
        self.noise = 60.0           # 底噪估计(自动跟随)
        self._stop = False
        self.level = 0.0            # 最近一块的音量, 给界面显示用
        self.thresh = ABS_MIN

    # ---------- 音频回调: 只入队, 不做任何耗时操作 ----------
    def _cb(self, indata, frames, t, status):
        self.blocks.put(indata[:, 0].copy())

    def _transcribe_and_send(self, pcm):
        try:
            text = voice_api.transcribe(pcm, rate=RATE)
        except Exception as e:
            print(f"[电脑麦] 识别异常: {e}")
            return
        if text:
            self.on_text(text)

    def run(self):
        if sd is None:
            print(f"[电脑麦] 打不开录音库(sounddevice): {_SD_ERROR}")
            return
        bs = int(RATE * BLOCK_SEC)
        try:
            with sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                                blocksize=bs, device=self.device,
                                callback=self._cb):
                print(f"[电脑麦] 已启动 16kHz, 直接说话即可 (设备: "
                      f"{sd.query_devices(self.device, 'input')['name']})")
                self._loop(bs)
        except Exception as e:
            print(f"[电脑麦] 打开麦克风失败: {e}")
            print("         检查: Windows 设置 -> 隐私 -> 麦克风 是否允许; 或插上耳麦")

    def _loop(self, bs):
        pre = []
        rec = []
        hot = 0
        quiet = 0
        talking = False
        dead_blocks = 0
        checked = False
        while not self._stop:
            try:
                blk = self.blocks.get(timeout=0.5)
            except queue.Empty:
                continue
            x = blk.astype(np.float64)
            rms = float(np.sqrt(((x - x.mean()) ** 2).mean()))
            self.level = rms

            # 前 30 块(3秒)全都是几乎为 0 -> 这台机器没实体麦克风, 提前说清楚
            if not checked:
                dead_blocks = dead_blocks + 1 if rms < 1.5 else 0
                if dead_blocks >= 30:
                    checked = True
                    print("[电脑麦] 警告: 麦克风一直是全 0, 说明这台电脑没有可用的实体麦克风")
                    print("         -> 插一个 3.5mm 耳麦 / USB 麦克风, 或在设备屏里选对设备")

            # 底噪跟随: 只在"不像人声"的时候更新, 速度放慢
            if rms < self.noise * 2.0:
                self.noise = self.noise * 0.95 + rms * 0.05
                if self.noise < 15.0:
                    self.noise = 15.0
            self.thresh = max(self.noise * 3.0, ABS_MIN)

            if not talking:
                pre.append(blk)
                if len(pre) > PRE_ROLL:
                    pre.pop(0)
                if rms > self.thresh:
                    hot += 1
                    if hot >= TRIGGER_BLOCKS:
                        talking = True
                        rec = list(pre)          # 把预录的 0.4 秒接上
                        quiet = 0
                        print(f"[电脑麦] 开始录 (音量 {rms:.0f} / 门槛 {self.thresh:.0f})")
                else:
                    hot = 0
            else:
                rec.append(blk)
                if rms > self.thresh * 0.9:
                    quiet = 0
                else:
                    quiet += 1
                long_enough = len(rec) * BLOCK_SEC >= MAX_SEC
                if quiet >= QUIET_BLOCKS or long_enough:
                    talking = False
                    hot = 0
                    pre = []
                    if len(rec) >= 3:            # 太短(0.3秒)当误触
                        pcm = np.concatenate(rec).astype(np.int16).tobytes()
                        print(f"[电脑麦] 录到 {len(rec) * BLOCK_SEC:.1f} 秒, 送去识别")
                        threading.Thread(target=self._transcribe_and_send,
                                         args=(pcm,), daemon=True).start()
                    rec = []

    def stop(self):
        self._stop = True


def start(on_text, device=None):
    """启动电脑麦克风通道(后台线程), 返回 PcMic 对象"""
    mic = PcMic(on_text, device=device)
    threading.Thread(target=mic.run, daemon=True).start()
    return mic


# ---- 单独运行 = 语音输入测试器 ----
if __name__ == "__main__":
    print("电脑麦克风测试: 对着麦克风说句话, 看能否识别。Ctrl+C 退出。\n")
    if sd is not None:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"  输入设备 {i}: {d['name']}")
        print()
    start(lambda t: print(f"识别结果 -> {t}"))
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("退出")
