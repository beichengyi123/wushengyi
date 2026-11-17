# -*- coding: utf-8 -*-
# ================= 百度语音转文字模块 =================
# 作用: 把 ESP32 发来的录音(PCM) 转成文字, 供主文件 1.py 调用
# 用法: 填好下面两个钥匙后, 在 1.py 里 import voice_api 即可
#       text = voice_api.transcribe(音频bytes)
#
# 钥匙获取步骤(需要你做, 约5分钟):
#   1. 浏览器打开 https://ai.baidu.com 注册/登录百度账号
#   2. 控制台 -> 产品服务/语音技术 -> 创建应用
#      技术选型勾"语音识别-短语音识别(标准版)", 其他随便, 描述随便填
#   3. 创建完在应用详情页能看到 API Key 和 Secret Key
#   4. 复制粘贴到下面两行引号里
# ====================================================

import base64
import json
import time
import urllib.request
import numpy as np

API_KEY = "在这里填入你的百度API_KEY"
SECRET_KEY = "在这里填入你的百度SECRET_KEY"

_token = {"value": "", "expire": 0}

def _get_token():
    """获取百度 access_token, 有效期约30天, 缓存起来不重复请求"""
    if _token["value"] and time.time() < _token["expire"]:
        return _token["value"]
    url = ("https://aip.baidubce.com/oauth/2.0/token"
           "?grant_type=client_credentials"
           f"&client_id={API_KEY}&client_secret={SECRET_KEY}")
    resp = json.loads(urllib.request.urlopen(url, timeout=10).read())
    if "access_token" not in resp:
        raise RuntimeError(f"获取token失败: {resp}")
    _token["value"] = resp["access_token"]
    _token["expire"] = time.time() + resp.get("expires_in", 2592000) - 3600
    return _token["value"]

def _preprocess(pcm_bytes, rate=8000):
    """去直流 + 300Hz 高通(实测电源160Hz嗡声占能量20%~54%, 人声被淹没, 必须滤掉)"""
    data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if data.size < 32:
        return data
    data = data - data.mean()                       # 去直流
    try:
        from scipy.signal import butter, sosfilt
        # 4阶 Butterworth 高通 300Hz: 男声基频(120~250Hz)会损失一部分,
        # 但共鸣/辅音信息全在 300Hz 以上, 识别可懂度反而大幅提升
        sos = butter(4, 300, btype="highpass", fs=rate, output="sos")
        data = sosfilt(sos, data)
    except ImportError:
        # 没装 scipy 时退回滑动均值法(约50Hz高通, 只去漂移)
        win = max(8, int(0.02 * rate))
        kern = np.ones(win, dtype=np.float32) / win
        drift = np.convolve(data, kern, mode="same")
        data = data - drift
    return data

def ac_level(pcm_bytes, rate=8000):
    """真实人声音量(交流有效值), 不含直流偏置 — 用来判断"到底说没说话\""""
    d = _preprocess(pcm_bytes, rate)
    if d.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(d ** 2)))

def _resample_8k_to_16k(pcm_bytes):
    """ESP32 录的是 8kHz, 百度要求 16kHz, 这里插值升采样"""
    data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    n_out = int(len(data) * 2)
    out = np.interp(np.linspace(0, len(data) - 1, n_out),
                    np.arange(len(data)), data)
    return out.astype(np.int16).tobytes()

def transcribe(pcm_bytes, retry=1, rate=8000):
    """输入: 8kHz 或 16kHz / 16bit / 单声道 PCM 原始字节
       rate: 这段音频的采样率(默认 8000 = ESP32; 电脑麦克风传 16000)
       返回: 识别出的文字(带标点), 失败返回空字符串
       retry: 网络类失败时的重试次数(默认 1 次), 不重复请求纯识别失败"""
    if not API_KEY or not SECRET_KEY:
        print("[voice_api] 还没填 API Key / Secret Key!")
        return ""
    d = _preprocess(pcm_bytes, rate)                # 去直流+去低频漂移
    if d.size == 0:
        return ""
    # 音量归一化: 把峰值拉到 60% 量程, 太小的声音也能被听清
    peak = float(np.abs(d).max())
    if peak > 1e-6:
        d = d * min(19660.0 / peak, 8.0)            # 最多放大 8 倍, 防止把噪声吹爆
    speech_pcm = np.clip(d, -32000, 32000).astype(np.int16).tobytes()

    for attempt in range(retry + 1):
        try:
            payload = json.dumps({
                "format": "pcm", "rate": rate, "channel": 1,
                "cuid": "esp32_hand", "token": _get_token(),
                "dev_pid": 1537,      # 普通话(支持简单英文)
                "pd": 1,              # 带标点, 输出"你好，今天天气不错。"更像人话
                "speech": base64.b64encode(speech_pcm).decode(),
                "len": len(speech_pcm),
            }).encode()
            req = urllib.request.Request(
                "http://vop.baidu.com/server_api", data=payload,
                headers={"Content-Type": "application/json"})
            resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
            if resp.get("err_no") == 0:
                text = resp["result"][0].strip() if resp.get("result") else ""
                return text
            print(f"[voice_api] 识别失败 err_no={resp.get('err_no')} {resp.get('err_msg')}")
            return ""
        except Exception as e:
            if attempt < retry:
                print(f"[voice_api] 网络异常, 重试一次: {e}")
                _token["expire"] = 0        # token 可能过期, 下次重新取
                continue
            print(f"[voice_api] 异常: {e}")
            return ""
    return ""

# ---- 直接运行本文件可以拿一段测试音频自测 ----
if __name__ == "__main__":
    print("本模块是给 1.py 调用的, 不单独运行。")
    print("钥匙已填:", "是" if API_KEY else "否 (去 https://ai.baidu.com 申请)")
