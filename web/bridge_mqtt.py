# -*- coding: utf-8 -*-
"""
无声有译 · MQTT -> HTTP 桥接 (不改 1.py 的那根线)

为什么需要它:
  1.py 只会把结果发 MQTT(给老 ESP32 用), 不会发 HTTP。
  不能改 1.py, 所以由本脚本"旁听":
    - 盯 sign_log.txt  -> 每个确认的词 + 置信度分数 + 距离 + 单双手 + 当前句
    - 订阅 MQTT        -> 屏幕上的文字(逐词拼接 / 整句 / 清屏) 和 语音识别文字
  两个来源合起来, POST 给 Flask 后端, 网页和 TFT 才有数据。

启动顺序: 1.py 先开 -> 再开本脚本 -> 再开 app.py(先后无所谓, 但 1.py 要在跑)
运行:   D:/python/python.exe web/bridge_mqtt.py
"""

import os
import re
import ast
import json
import time
import threading
import urllib.request

import paho.mqtt.client as mqtt

# ================= 配置(必须与 1.py 顶部一致, 改一处就要两边一起改) =================
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_SIGN_TEXT = "aiot/sign/gesture/demo"    # 1.py 发给屏幕的手语文字
TOPIC_AUDIO_TEXT = "aiot/voice/text"          # 1.py 发给屏幕的语音文字

FLASK_URL = "http://127.0.0.1:5000"           # 本机后端

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)          # 无声有译-项目
LOG_PATH = os.path.join(PROJ_DIR, "sign_log.txt")   # 1.py 的运行日志

FRESH_SEC = 3.0     # 日志里的分数, 超过这么久就不再配给 MQTT 消息
# ==============================================================================

# 日志行格式:  22:11:57 叫 d=7.3 门槛8.0 分数0.92 单手 本句=['叫', '我']
LOG_RE = re.compile(
    r"^(\d{2}:\d{2}:\d{2})\s+(\S+)\s+d=([\d.]+)\s+门槛([\d.]+)\s+分数([\d.]+)\s+"
    r"(单手|双手)\s+本句=\[(.*)\]\s*$"
)

# 共享上下文: MQTT 线程要靠它判断"这条消息是不是某个词的上屏"
_ctx = {
    "sentence": [],     # 最新一行日志里的"本句"词表
    "score": None,      # 最新一行日志的分数
    "word": "",
    "at": 0.0,
}
_ctx_lock = threading.Lock()


def post(path, payload, retries=2):
    """发给后端; 后端没开时不要崩, 打一行提示继续跑"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(
                FLASK_URL + path, data=data,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if i == retries:
                print(f"[桥接] 发给后端失败({path}): {e}  <- 检查 app.py 是否在运行")
            else:
                time.sleep(0.3)
    return None


def tail_log():
    """跟 tail -f 一样读 sign_log.txt; 从文件末尾开始, 不重放历史"""
    print(f"[桥接] 开始盯日志: {LOG_PATH}")
    while not os.path.exists(LOG_PATH):
        print("[桥接] 日志还没生成(1.py 说过'每次确认一个词记一行', 先比几个词就会有了)")
        time.sleep(2)
    fp = open(LOG_PATH, "r", encoding="utf-8", errors="replace")
    fp.seek(0, os.SEEK_END)
    while True:
        line = fp.readline()
        if not line:
            time.sleep(0.2)
            continue
        m = LOG_RE.match(line.strip())
        if not m:
            continue
        hhmmss, word, dist, th, score, hand, sent = m.groups()
        try:
            sentence = ast.literal_eval("[" + sent + "]") if sent.strip() else []
        except Exception:
            sentence = []
        with _ctx_lock:
            _ctx.update(sentence=list(sentence), score=float(score), word=word, at=time.time())
        r = post("/api/sign_word", {
            "word": word, "dist": float(dist), "th": float(th),
            "score": float(score), "hand": hand,
            "sentence": " + ".join(sentence),
            "ts": time.strftime("%Y-%m-%d ") + hhmmss,
        })
        flag = " [低于阈值, 不推TFT]" if (r and r.get("blocked")) else ""
        print(f"[手语] {word:<4} 分数{score} {hand} 本句:{' + '.join(sentence) or '(空)'}{flag}")


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[桥接] MQTT 已连接 {MQTT_BROKER}")
        client.subscribe(TOPIC_SIGN_TEXT)
        client.subscribe(TOPIC_AUDIO_TEXT)
        print(f"[桥接] 已订阅: {TOPIC_SIGN_TEXT} / {TOPIC_AUDIO_TEXT}")
    else:
        print(f"[桥接] MQTT 连接失败 rc={rc} (检查网络能不能连上 broker)")


def on_message(client, userdata, msg):
    topic = msg.topic
    text = msg.payload.decode("utf-8", errors="replace")

    if topic == TOPIC_AUDIO_TEXT:
        if text.strip():
            post("/api/display", {"source": "voice", "text": text,
                                  "event": "voice", "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
            print(f"[语音] {text}")
        return

    # ---- 手语文字: 分清 清屏 / 逐词更新 / 整句 ----
    if not text.strip():
        post("/api/display", {"source": "sign", "text": "", "event": "clear", "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
        print("[屏幕] 清空")
        return

    with _ctx_lock:
        sentence = list(_ctx["sentence"])
        score = _ctx["score"] if (time.time() - _ctx["at"]) <= FRESH_SEC else None

    joined = "".join(sentence)[:18]
    if sentence and text == joined:
        ev, sc = "word", score
    else:
        # 整句上屏(Enter / 撤销后重发): 词已各自过过阈值, 不再按单词语义拦
        ev, sc = "final", None

    post("/api/display", {"source": "sign", "text": text, "event": ev,
                          "score": sc, "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"[屏幕-{ev}] {text}")


def main():
    threading.Thread(target=tail_log, daemon=True).start()

    c = mqtt.Client()
    c.on_connect = on_connect
    c.on_message = on_message
    c.reconnect_delay_set(min_delay=1, max_delay=10)
    print(f"[桥接] 正在连接 {MQTT_BROKER}:{MQTT_PORT} ...")
    c.connect(MQTT_BROKER, MQTT_PORT, 60)
    c.loop_forever()     # 断线会自动重连


if __name__ == "__main__":
    main()
