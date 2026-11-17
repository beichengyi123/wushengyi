# -*- coding: utf-8 -*-
# ================= 无声有译 主文件 =================
# 手语通道: MediaPipe 双手关键点 + kNN 真模型 (sign_model.npz)
# 语音通道: 收 ESP32 音频 -> 暂为假 AI (接百度 API 后替换)
# 屏幕显示: 全部通过 MQTT 发给 ESP32, 固件不用改
# ==================================================
import os
import re
import sys
import cv2
import numpy as np
import mediapipe as mp
import paho.mqtt.client as mqtt
import threading
import time
from PIL import Image, ImageDraw, ImageFont
import voice_api   # 百度语音转文字模块 (同目录 voice_api.py)
import sentence_api  # 手语词 -> 整句 (同目录 sentence_api.py)

# 防误检 + 手位校验（与采集页/自检共用同一套规则, 见 sign_guard.py）
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import sign_guard  # noqa: E402

# ================= 配置区 =================
# 注: broker.emqx.io 2026-10-26 当天大面积超时, 换到同样免费的 HiveMQ 公共服务器
MQTT_BROKER = "broker.hivemq.com"
TOPIC_SIGN_TEXT  = "aiot/sign/gesture/demo"   # 手语文字主题
TOPIC_AUDIO_SEND = "aiot/voice/audio"         # ESP32发音频的主题
TOPIC_AUDIO_TEXT = "aiot/voice/text"          # 电脑回传文字的主题

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sign_model.npz")

K = 3                # kNN 邻居数
STABLE_NEED = 4      # 连续认对多少个窗口才确认
FIXED_SEQ = 10       # 必须和 train_model.py 一致
RECORD_FRAMES = 45   # 必须和 sign_collect.py 一致
MOTION_W = 0.7       # 必须和 train_model.py 一致
DIST_THRESHOLD = 4.0 # 词没有门槛时的兜底值

SENTENCE_IDLE = 3.5  # 最后一个词之后停顿多久(秒)就认为这句打完了, 交给语言模型组句
CONT_WINDOW = 8.0    # 上屏后这么久内又识别到新词, 算"这句还没说完", 合并进上一句重发
MAX_WORDS = 8        # 一句话最多攒几个词
SHOW_WORD_LIVE = False  # False=单词不上屏(攒着等组句); True=每个词立刻上屏(调试用)
WORD_GAP = 1.5       # 同一个词最快多久记一次(秒), 防手一直比着被反复记

# 语音从哪来: "esp32"=板子上的麦克风(8kHz); "pc"=电脑/耳麦麦克风(16kHz, 音质好得多)
# 2026-10-28 复测: ESP32 链路宽带正常(95%能量截止 2770~3621Hz), 已切回 esp32
# (当时判"只剩500Hz"其实是板子断网期间的残包, 线没坏)
VOICE_SOURCE = "esp32"
VOICE_PC_DEVICE = None   # None=系统默认麦克风; 插了耳麦想指定设备就填设备号(运行可看列表)
# ==========================================

MQTT_KEEPALIVE = 30      # 心跳 30 秒(默认 60): 云端不至于把空闲连接判死
CLIENT_ID_PREFIX = "PC_WuSheng_"   # 加时间戳后缀: 同名 ID 多开会互相踢下线
_AUDIO_CB = None         # 语音回调, 重建客户端时要重新挂上
_MQTT_LOCK = threading.Lock()   # 看门狗和发布线程会同时想重连, 不串行化会互相踢
client = None


def _on_mqtt_connect(c, userdata, flags, rc):
    """每次(重新)连上都补订阅音频主题: 掉线重连后订阅不会丢"""
    print(f"[MQTT] 通道就绪 (rc={rc})")
    if rc == 0:
        try:
            c.subscribe(TOPIC_AUDIO_SEND)
        except Exception:
            pass


def _new_client():
    c = mqtt.Client(CLIENT_ID_PREFIX + str(int(time.time() * 1000) % 1000000))
    c.reconnect_delay_set(min_delay=1, max_delay=10)
    c.on_connect = _on_mqtt_connect
    if _AUDIO_CB is not None:
        c.on_message = _AUDIO_CB
    return c


def _mqtt_connect(wait=5.0):
    """(重新)建立连接并等它真的连上(CONNACK 后才算数)"""
    global client
    with _MQTT_LOCK:
        if client is not None and client.is_connected():
            return True
        old = client
        if old is not None:
            try:
                old.loop_stop()
            except Exception:
                pass
            try:
                old.disconnect()
            except Exception:
                pass
        client = _new_client()
        try:
            client.connect(MQTT_BROKER, 1883, keepalive=MQTT_KEEPALIVE)
            client.loop_start()      # 后台线程跑网络循环: 没有它 60 秒后会被静默掉线
        except Exception as e:
            print(f"[MQTT] 连接失败({e}), 稍后自动重试(检查网络或WiFi)")
            return False
        deadline = time.time() + wait
        while time.time() < deadline:
            if client.is_connected():
                print(f"MQTT 已连接: {MQTT_BROKER}")
                return True
            time.sleep(0.1)
        print("[MQTT] 连接超时(网络慢或被墙), 稍后重试")
        return False


def _mqtt_online(wait=5.0):
    """确保连接真的可用; 不可用就重建一个(主动断开/被云端踢掉都能救回来)"""
    if client is not None and client.is_connected():
        return True
    return _mqtt_connect(wait)


def _publish(topic, text, retries=3):
    """统一发布出口: 保证在线 -> QoS1 -> 等 PUBACK 确认, 确认不了才算失败"""
    payload = (text if text and str(text).strip() else " ").encode("utf-8")
    for i in range(retries):
        try:
            if not _mqtt_online():
                print(f"[MQTT] 仍不在线, 重试({i + 1}/{retries})")
                time.sleep(0.8)
                continue
            c = client
            if c is None:
                time.sleep(0.8)
                continue
            info = c.publish(topic, payload, qos=1)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                print(f"[MQTT] 发布返回码 {info.rc}, 重试({i + 1}/{retries})")
                time.sleep(0.8)
                continue
            try:
                info.wait_for_publish(timeout=3)
            except Exception:
                pass
            if info.is_published():
                return True                      # 云端已确认收到
            print(f"[MQTT] 等不到确认, 重试({i + 1}/{retries})")
        except Exception as e:
            print(f"[MQTT] 发布异常({e}), 重试({i + 1}/{retries})")
        time.sleep(0.8)
    print(f"[MQTT] 上屏失败, 屏幕没收到: {text!r} (检查网络/MQTT 服务器)")
    return False


def publish_sign(text, retries=3):
    """把文字发到 TFT 屏(统一出口)。掉线/云端踢线都会自动重连再发,
       不会再出现"界面在变、屏幕不动"的静默失败"""
    return _publish(TOPIC_SIGN_TEXT, text, retries)


def publish_audio_text(text, retries=2):
    """语音转写结果发到 TFT 屏(同一个统一出口, 只是换个主题)"""
    return _publish(TOPIC_AUDIO_TEXT, text, retries)


def _mqtt_watchdog():
    """后台看门狗: 每 10 秒看一眼, 掉线就悄悄连回来(不打扰用户)"""
    while True:
        time.sleep(10)
        try:
            if not client.is_connected():
                print("[MQTT] 检测到掉线, 正在重连...")
                if _mqtt_online():
                    print("[MQTT] 已重连")
        except Exception as e:
            print(f"[MQTT] 重连失败: {e} (10 秒后再试)")


_mqtt_connect()                      # 首次连接: 失败也不阻塞启动, 后面会自动补
threading.Thread(target=_mqtt_watchdog, daemon=True).start()


# ================= 真模型加载 =================
model = np.load(MODEL_PATH, allow_pickle=False)
MX, My = model["X"], model["y"]
_WORDS = [str(w) for w in model["thresh_words"].tolist()]

# ---- 每词参数: 全部是从 sign_data/<词名>/ 那几十条样本里算出来的, 不是拍脑袋 ----
#   th  : 双手状态下, 实时比划离这个词多远还算"是它"
#   th1 : 只检测到一只手时用的门槛(单手/双手两套样本、两套门槛)
#   ty  : 典型距离 = 这个词自己每遍之间本来差多少。
#         识别时拿它当分母 (d / ty), 让"自己就比较散"的词不吃亏。
#   two : 这个词是不是必须两只手都进画面才参与判断
TH = dict(zip(_WORDS, model["thresh_vals"].tolist()))
TH1 = dict(zip(_WORDS, model["thresh1_vals"].tolist())) if "thresh1_vals" in model else dict(TH)
TY = dict(zip(_WORDS, model["typical_vals"].tolist())) if "typical_vals" in model else {}
TY1 = dict(zip(_WORDS, model["typical1_vals"].tolist())) if "typical1_vals" in model else dict(TY)
TY1 = {w: TY1.get(w, TY.get(w, 4.0)) for w in _WORDS}

# 上一版这里做过"单手变体匹配"(把样本的第二只手也清零再比), 结果是
# 只做一半动作也能过门槛 —— 双手词"现在"单手就能被识别出来, 这是错的。
# 现在是硬规则: 双手词必须两只手都进画面, 否则直接不进候选。
TWO = ({w: bool(v) for w, v in zip(_WORDS, model["two_hand_vals"].tolist())}
       if "two_hand_vals" in model else {})

# ---- 特殊词: 录样本时踩过的坑, 钉在这里, 下次重训/换个词库都不用重新想一遍 ----
# 2026-10-28「学校」: 旧样本是"手型对、但两手高低差不够", 20 条散度中位 2.5,
#   和名字(4.0)尺度接近, 于是两个词老是互换。删掉整目录重录 20 条之后:
#   散度中位 1.9 / p97 4.7, 到名字最近的样本距离 13.3,
#   相对分数 学校 13.3/1.9=7.0 vs 名字 16/4.0=4.0, 判名字; 学校自己 13 以内判学校。
#   => 这一条属于"样本自己把问题解决了", 代码不用特殊照顾, 记在这只为留个证据。
#   剩下的老问题: 名字/学校都还挂着双手门槛下限 20.0, 接受圈比它们自己的散度
#   宽一个量级, 所以库里没有的手势偶尔会被硬塞成这两个词之一 —— 要收紧就在
#   SPECIAL 里单独给它俩调 th, 别动全局下限(动了"现在""帮助"就全哑了)。
# 手改格式(只写想改的键, 没写的继续跟模型走):
#   SPECIAL = {"学校": {"th": 18.0}, "名字": {"th1": 9.0}}
# 2026-10-29「帮助」: 旧样本 20 条里有 12 条是"录到一半第二只手丢了"的残缺样本,
#   把自身散度 p97 撑到 11.3 -> 门槛跟着涨到 25.8, 它的接受圈比所有词都宽,
#   于是别的词在只检测到一只手时容易被它吸走(历史 bug: 学校 -> 帮助)。
#   整目录删掉重录 20 条(这次 20 条双手帧全部完整)之后实测:
#     自身 p97 11.3 -> 2.5, 门槛 25.8 -> 20.0(已到双手下限),
#     到最近异类词的距离 19.3, 不再盖住学校的 16.3。
#   回归(loo_check.py / mask_hand_check.py / test_live.py):
#     留一验证 99.4%(和重录前持平); 单手场景 99% —— 双手词在只看到一只手时
#     本来就被排除(下面 _live_classify 里的 allow 那段), 所以"再见被判成帮助"
#     只在纯 kNN 里出现, 实时判定层不会发生, 不用为此收紧 th1;
#     "我要帮助你" 整句回放全对。
#   => 同样是"样本自己把问题解决了", 代码不用特殊照顾, 记在这只为留个证据。
SPECIAL = {}

def _apply_special():
    for w, kv in SPECIAL.items():
        if w not in TH:
            print(f"[警告] SPECIAL 里的词「{w}」不在模型词库里, 已忽略")
            continue
        if "th" in kv:
            TH[w] = float(kv["th"])
        if "th1" in kv:
            TH1[w] = float(kv["th1"])
        if "ty" in kv:
            TY[w] = float(kv["ty"])
        if "ty1" in kv:
            TY1[w] = float(kv["ty1"])
        if "two" in kv:
            TWO[w] = bool(kv["two"])

_apply_special()
SEG_DIM = FIXED_SEQ * 126

MY_S = np.array([str(v) for v in My.tolist()])
WORD_IDX = {w: np.where(MY_S == w)[0] for w in np.unique(MY_S)}
SCORE_MAX = 3.0   # 相对分数上限: 距离超过该词典型距离 3 倍, 就当"不是我库里的词"

def print_word_table():
    """启动时把每个词的参数打出来: 判错了先看这张表, 不用猜模型里到底是什么数"""
    print(f"手语模型已加载: {len(My)} 条样本, 词库 {len(_WORDS)} 个")
    print(f"  {'词':<4}{'手':>3}{'门槛':>7}{'单手门槛':>9}{'典型':>7}{'单手典型':>9}")
    for w in _WORDS:
        mark = "  ← 手改" if w in SPECIAL else ""
        print(f"  {w:<4}{('双' if TWO.get(w) else '单'):>3}"
              f"{TH[w]:>7.1f}{TH1[w]:>9.1f}{TY.get(w, 4.0):>7.1f}"
              f"{TY1.get(w, TY.get(w, 4.0)):>9.1f}{mark}")

# ---- 运行日志: 每次确认一个词记一行 ----
# 事后能直接看出"判成了什么/距离多少/候选有谁", 不用凭感觉猜
LOG_PATH = os.path.join(BASE_DIR, "sign_log.txt")

def _log(s):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as fp:
            fp.write(f"{time.strftime('%H:%M:%S')} {s}\n")
    except Exception:
        pass

# ================= 中文字体 =================
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 34)
small_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)

def draw_cn(frame, text, pos, ft=font, color=(0, 255, 0)):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img).text(pos, text, font=ft, fill=color)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ================= 特征管线 (与采集/训练/验收脚本完全一致) =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

def frame_hands(result, rgb=None):
    """取一帧里两只手的原始坐标: (right_pts, left_pts), 缺哪只就是 None

    新增两道防误检闸门(不改识别本身, 只决定哪些"手"算数):
      · 手部置信下限: handedness 分数太低的检出直接丢掉
      · 人脸区域: 落在人脸框里的手骨架丢掉 —— MediaPipe 常把下巴/嘴角/耳朵
        误当成手, 这些假手一旦进缓冲, 就会凭空多出一个词
    rgb 传进来才有用(人脸检测要彩图); 不传就只做置信度过滤。"""
    boxes = sign_guard.face_boxes(rgb) if rgb is not None else []
    right, left, dropped = sign_guard.pick_hands(result, boxes)
    if right is None and left is None:
        return None
    return (right, left)

def _normalize_seq(frames, anchor_pref):
    """按锚点偏好组装 (T,126) 特征: 锚点手归一化, B手降噪, 去z"""
    out = []
    for right, left in frames:
        anchor, other = (right, left) if anchor_pref == "Right" else (left, right)
        if anchor is None and other is None:
            return None
        if anchor is None:
            anchor, other = other, None
        origin = anchor[0]
        scale = np.linalg.norm(anchor[9] - origin)
        if scale < 1e-6:
            return None
        a = ((anchor - origin) / scale).flatten()
        b = ((other - origin) / scale).flatten() if other is not None else np.zeros(63)
        out.append(np.concatenate([a, b]))
    arr = np.array(out, dtype=np.float32)
    b3 = arr[:, 63:].reshape(len(arr), 21, 3).copy()
    wrist = b3[:, 0:1, :]
    centered = b3 - wrist
    centered[:, 0, :] = wrist[:, 0, :] * 0.4
    arr = np.concatenate([arr[:, :63], centered.reshape(len(arr), 63)], axis=1)
    arr[:, 2::3] = 0
    return arr

def downsample(arr, n=FIXED_SEQ):
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return arr[idx].flatten()

def finalize(arr):
    """静态轨迹 + 运动特征, 与 train_model.py 完全一致"""
    ds = downsample(arr)
    seq = ds.reshape(FIXED_SEQ, -1)
    delta = np.zeros_like(seq)
    delta[1:] = seq[1:] - seq[:-1]
    return np.concatenate([ds, delta.flatten() * MOTION_W])

# ================= 功能 1.5：词 -> 整句 =================
# 手语是一个词一个词打出来的: 单词先攒在 buffer 里不上屏,
# 停顿够久(说明这句打完了)才交给 sentence_api 组装成句子, 一次性发屏幕。
# 这样屏幕上是"一句话", 而不是一个词一个词往外蹦。
_word_buf = []
_word_lock = threading.Lock()
_last_word_time = 0.0
_last_sent_words = []    # 上一句已经上屏的词 (用于"续句": 后面又来了词就接着上一句长)
_last_sent_at = 0.0

def add_sign_word(word):
    """收一个识别词(不上屏): 3 秒内的重复词只算一次, 避免同一个动作被记两遍"""
    global _last_word_time
    now = time.time()
    with _word_lock:
        if _word_buf and word == _word_buf[-1] and now - _last_word_time < 3.0:
            _last_word_time = now
            return False
        _word_buf.append(word)
        if len(_word_buf) > MAX_WORDS:
            _word_buf.pop(0)
        _last_word_time = now
        return True

def buf_snapshot():
    """给摄像头窗口显示用: 当前已攒到哪些词(已经过合并纠错)"""
    with _word_lock:
        return polish_words([list(h) for h in _word_buf])

def clear_session():
    """按 c 手动新起一句: 清空没上屏的词, 也不再和上一句合并"""
    global _last_sent_words, _last_sent_at, _last_word_time
    with _word_lock:
        _word_buf.clear()
        _last_sent_words = []
        _last_sent_at = 0.0
        _last_word_time = 0.0
    print("[清空] 下一句从零开始")

def undo_last_word():
    """按 z 撤销一个词(识别错了用):
       优先撤还没上屏的; 若都已上屏, 就把上句最后一个词删掉, 组句重发修正"""
    global _last_sent_words, _last_sent_at
    removed, rescreen = None, False
    with _word_lock:
        if _word_buf:
            removed = _word_buf.pop()[0]
            line = polish_words([list(h) for h in _word_buf])
            print(f"[撤销] 删掉: {removed} | 本句剩余: {line}")
            publish_sign("".join(line)[:18])
            return
        if _last_sent_words:
            removed = _last_sent_words.pop()
            _last_sent_at = time.time()   # 刷新, 保持续句窗口还能继续接
            rescreen = True
    if not removed:
        print("[撤销] 没有可撤销的词")
        return
    if rescreen and _last_sent_words:
        sentence = sentence_api.compose(list(_last_sent_words))
        publish_sign(sentence[:18])
        print(f"[撤销] 从上句删掉: {removed} -> 屏幕更正为: {sentence}")
    else:
        print(f"[撤销] 从上句删掉: {removed}, 句子已清空")

def sentence_worker():
    """攒到词 + 停顿 SENTENCE_IDLE 秒 -> 组装成句 -> 发屏幕
       若上屏后不久又来了新词, 视为"这句还没打完", 合起来重新组句再发一次
       (这样"我+谢谢+你"先出, 之后做"再见"会自动补成"我谢谢你, 再见。")"""
    global _word_buf, _last_sent_words, _last_sent_at
    while True:
        time.sleep(0.3)
        now = time.time()
        with _word_lock:
            if not _word_buf:
                continue
            if now - _last_word_time < SENTENCE_IDLE:
                continue
            words = polish_words([list(h) for h in _word_buf])
            _word_buf = []

        # 续句: 上一句刚上屏不久, 新词接着它往下说
        if _last_sent_words and now - _last_sent_at < CONT_WINDOW:
            merged = _last_sent_words + words
            print(f"[续句] 接上一句: {' + '.join(_last_sent_words)} "
                  f"+ {' + '.join(words)}")
        else:
            merged = words
        if len(merged) > MAX_WORDS:
            merged = merged[-MAX_WORDS:]

        # 单词不用惊动语言模型, compose 会原样返回
        sentence = sentence_api.compose(merged)
        if not sentence:
            continue
        # 屏幕一行约 18 个汉字, 超了截断
        show = sentence[:18]
        publish_sign(show)
        _last_sent_words, _last_sent_at = merged, time.time()
        print(f"📝 [成句] {' + '.join(merged)} -> {sentence} -> 已发屏幕")



# ================= 功能 1a：录播识别（边做边存, 按 Enter 一次性识别整句） =================
# 为什么改成这样:
#   老的实时识别每个词要走"滑动窗口每3帧比对 -> 连续4窗认对 -> 词间隔1.5秒
#   -> 缓冲清空重攒45帧", 一个词实际要 3 秒以上。人比一句话时词与词只隔
#   0.2~0.5 秒, 后面的动作必然被吞掉("我现在要去学校"只剩"我现在要")。
#   现在改成: 做动作时只存帧不认, 按 Enter 后把整段切段、逐段分类, 一次性出句。
#   识别速度和人比划的速度彻底解耦 —— 比划多快都行, 算多慢都不丢词。
REC_MODE = False          # 已弃用: 录播(整段一次性识别)模式, 相关函数留着给离线测试用
                          # 现在主循环走的是"实时出词 + Enter 输出"
MAX_REC_FRAMES = 60 * 30  # 最多缓存 60 秒动作, 超了自动丢最早的
MIN_SEG = 10              # 少于 10 帧(约0.3秒)的片段不认: 太短不可能是完整动作
MAX_SEG = 75              # 超过 75 帧(约2.5秒)的片段要拆开, 里面不止一个词
SEG_SILENCE = 4           # 手离开画面连续 4 帧以上 = 两个词之间的分界
SCAN_WINS = (16, 24, 32, 45, 60)  # 候选窗口长度(帧): 程序自己挑最贴合实际比划速度的
                          # 比划得快 或 摄像头帧率低 就选16/24, 慢就选60
                          # (帧率低时同样 1 秒动作只有十几帧, 窗口必须跟着变小)
_fps = 0.0                # 实测摄像头帧率(显示+识别用)

_rec = []                 # 录下的帧: (right, left) 或 None(这一帧没手)
_rec_lock = threading.Lock()
_last_words = []          # 上次识别出的词序列 (z 撤销时用)
_batch_busy = False       # 正在批量识别, 防止连按 Enter 重复跑


def _rec_clear():
    with _rec_lock:
        _rec.clear()


def _classify_seq(arr):
    """(T,126) 已归一化序列 -> (词, 距离)。判定规则和实时模式完全一致"""
    f = finalize(arr)
    one_hand = not bool(np.any(arr[:, 63:]))
    d = np.linalg.norm(MX - f, axis=1)
    if one_hand and TWO:
        allow = np.ones(len(My), dtype=bool)
        for _w, _ix in WORD_IDX.items():
            if TWO.get(_w, False):
                allow[_ix] = False
        if not allow.any():
            return None, 1e9, one_hand
        d = np.where(allow, d, np.inf)
    top = np.argsort(d)[:K]
    votes = [str(My[j]) for j in top]
    uniq = []
    for v in votes:
        if v not in uniq:
            uniq.append(v)

    def _score_of(w):
        dd = float(np.sort(d[WORD_IDX[w]])[:3].mean())
        return dd / max(float(TY.get(w, 4.0)), 0.8)

    if len(uniq) >= 2 and votes.count(uniq[0]) - votes.count(uniq[1]) <= 1:
        uniq.sort(key=_score_of)
    pred = uniq[0]
    same = [d[j] for j in top if str(My[j]) == pred]
    mean_d = float(np.mean(same)) if same else float(d[top[0]])
    return pred, mean_d, one_hand


def _classify_frames(seg):
    """一段原始关键点 -> (词, 距离, 单手?) 两种锚点各试一遍, 取更近的"""
    best = None
    for pref in ("Right", "Left"):
        arr = _normalize_seq(seg, pref)
        if arr is None or len(arr) < 3:
            continue
        pred, dd, oh = _classify_seq(arr)
        if pred is None or not np.isfinite(dd):
            continue
        if best is None or dd < best[1]:
            best = (pred, dd, oh)
    return best


def _blocks(frames, silence=SEG_SILENCE):
    """整段录音 -> 若干连续块: 手没离开画面就一直算同一块"""
    out, cur, gap = [], [], 0
    for f in frames:
        if f is None:
            gap += 1
            if cur and gap >= silence:
                out.append(cur)
                cur = []
            continue
        gap = 0
        cur.append(f)
    if cur:
        out.append(cur)
    return [b for b in out if len(b) >= MIN_SEG]


def _picks(block, W, step=3):
    """用一个固定长度 W 的窗口扫一遍, 返回分数最低的几个位置(峰)
       分数 = 距离 / 该词自己的门槛, 越小越像; 相邻峰至少隔 0.55*W 帧"""
    if len(block) < W:
        r = _classify_frames(block)
        if r is None:
            return []
        pred, dd, oh = r
        th = float(TH.get(pred, DIST_THRESHOLD))
        return [(dd / max(th, 1e-6), 0, pred, dd, th, oh)] if dd < th else []
    cands = []
    for st in range(0, len(block) - W + 1, step):
        r = _classify_frames(block[st:st + W])
        if r is None:
            continue
        pred, dd, oh = r
        th = float(TH.get(pred, DIST_THRESHOLD))
        cands.append((dd / max(th, 1e-6), st, pred, dd, th, oh))
    cands.sort(key=lambda c: c[0])
    picked = []
    sup = max(10, int(W * 0.55))
    for c in cands:
        if c[0] > 1.05:              # 门槛都过不了, 后面的只会更差
            break
        if any(abs(c[1] - p[1]) < sup for p in picked):
            continue                 # 离已选位置太近 = 同一个动作, 跳过
        picked.append(c)
    picked.sort(key=lambda c: c[1])
    return picked


def _scan_block(block):
    """一块里可能是一个词, 也可能是连着打的几个词。
    先给同一块试几种窗口长度(SCAN_WINS), 每种粗扫一遍, 选"词数×窗长最贴合
    块长、且分数低"的那个长度, 再用它细扫。
    为什么必须这样: 窗口比实际动作长 -> 一个窗里装两个词, 混; 比实际动作
    短 -> 只看到一半动作。人比划快慢差一倍很常见, 固定窗长会崩一边。"""
    if len(block) < 20:
        return []
    best = None
    for W in SCAN_WINS:
        if len(block) < W * 0.7:
            continue
        p = _picks(block, W, step=8)      # 粗扫定窗长
        if not p:
            continue
        avg = float(np.mean([c[0] for c in p]))
        mis = abs(len(p) * W - len(block)) / len(block)   # 窗口铺满这块的程度
        q = avg + 0.8 * mis
        if best is None or q < best[0]:
            best = (q, W)
    if best is None:
        return []
    return [(c[2], c[3], c[4], c[5]) for c in _picks(block, best[1], step=3)]


def batch_recognize(frames):
    """整段 -> (词序列, 每块明细, 耗时)"""
    words, details = [], []
    t0 = time.time()
    for k, b in enumerate(_blocks(frames)):
        picks = _scan_block(b)
        if not picks:
            details.append((k + 1, len(b), "—", 0.0, 0.0, False, False))
            continue
        for pred, dd, th, oh in picks:
            details.append((k + 1, len(b), pred, dd, th, True, oh))
            if words and words[-1] == pred:     # 同一动作被记两次: 只算一次
                continue
            words.append(pred)
    return words, details, time.time() - t0

def flush_recognition():
    """按 Enter: 把录下来的整段识别成词 -> 组句 -> 发屏幕, 然后清空重录"""
    global _batch_busy, _last_words
    if _batch_busy:
        print("[识别] 上一轮还在算, 稍等")
        return
    with _rec_lock:
        frames = list(_rec)
        _rec.clear()
    if not frames:
        print("[识别] 还没录到动作 (先比划再按 Enter)")
        return
    _batch_busy = True

    def _work():
        global _batch_busy, _last_words
        try:
            # 存一份原始动作帧: 事后能拿真实数据复盘/调参, 不用重录
            try:
                _dump_dir = os.path.join(BASE_DIR, "sign_live")
                os.makedirs(_dump_dir, exist_ok=True)
                _tag = time.strftime("%m%d_%H%M%S")
                np.save(os.path.join(_dump_dir, f"rec_{_tag}.npy"),
                        np.array(frames, dtype=object), allow_pickle=True)
            except Exception as _e:
                print(f"[存档] 失败: {_e}")
            words, details, cost = batch_recognize(frames)
            print(f"=== 整段识别: 录了{len(frames)/max(_fps,1.0):.1f}秒 ({len(frames)}帧 "
                  f"@{_fps:.0f}fps) / {len(_blocks(frames))}块 / 认出{len(words)}词 / "
                  f"计算{cost:.2f}秒 ===")
            for no, n, pred, dd, th, ok, oh in details:
                if not ok:
                    print(f"   块{no:<2} {n:>3}帧 -> 这段不像任何词, 跳过")
                    continue
                print(f"   块{no:<2} {n:>3}帧 -> {pred:<4} 距离{dd:5.1f}/门槛{th:5.1f} ✓ "
                      f"{'单手' if oh else '双手'}")
            if not words:
                print("⚠️ 整段没认出可用词 (动作幅度太小 或 手势不在词库里)")
                return
            print(f"🤟 识别结果: {' + '.join(words)}")
            sentence = sentence_api.compose(list(words))
            show = (sentence or "".join(words))[:18]
            publish_sign(show)
            _last_words = list(words)
            print(f"📝 已发屏幕: {show}")
        finally:
            _batch_busy = False

    threading.Thread(target=_work, daemon=True).start()


def undo_last_words():
    """按 z: 删掉上一句结果的最后一个词, 重新组句上屏(识别错了用)"""
    global _last_words
    if not _last_words:
        print("[撤销] 还没有可撤销的结果")
        return
    removed = _last_words.pop()
    if _last_words:
        sentence = sentence_api.compose(list(_last_words))
        show = (sentence or "".join(_last_words))[:18]
        publish_sign(show)
        print(f"[撤销] 删掉「{removed}」 -> 屏幕: {show}")
    else:
        publish_sign("")
        print(f"[撤销] 删掉「{removed}」, 已清屏")


# ================= 功能 1：连续流式手语识别（后处理层） =================
# ------------------------------------------------------------------
# ⚠️ 这一层只做"后处理", 不碰底层:
#    摄像头 -> MediaPipe 取 21 点关键点      (原基础代码, 未改动)
#          -> _normalize_seq/finalize 特征   (原基础代码, 未改动)
#          -> _classify_rank/_live_classify  kNN 分类 (原基础代码, 未改动)
#          -> 【本层】LiveRecognizer  滑窗 + 防抖 + 稳定/切换判定
#          -> 【本层】polish_words()  词组合并 + 纠错过滤
#          -> 上屏 / MQTT
#    连贯手语的关键: 每来一帧就滑窗判一次, 同一个手势"稳住"够了才出词,
#    换了手势才更新结果 —— 不需要打完一个手势停下来等识别。
# ------------------------------------------------------------------

# ================= 参数区: 想调效果改这里就行, 改完重启 1.py 生效 =================
# 下面这批是你手调的值, 都按 30fps 标定。程序每秒会按摄像头实测帧率自动换算一次。
# 改 *_REF 就行; 下面派生出来那几个会自动被 apply_fps() 覆盖, 不用手动改。
FPS_REF = 30.0
LIVE_WINS_REF = (16, 22, 30, 45)  # 判词窗口(帧)候选: 快比划选短的, 慢比划选长的
                                  # 试过再加 12 帧窗口来照顾"飞快比划": 0.5秒一个词时能救回来,
                                  # 但代价是手随便动一下就被误认成词(多词数从2个涨到11个), 不划算, 已放弃。
                                  # 所以正常比划节奏请保证每个动作 0.6 秒以上。
JUDGE_EVERY_REF = 2               # 每几帧判一次。1=每帧判(最灵敏, 吃CPU); 2=隔帧(推荐)
JUDGE_WINDOW_REF = 5              # 滑窗宽度: 用最近这么多次判定做多数表决
STABLE_VOTES_REF = 3              # 稳定判定: 窗内同一个词至少得这么多票才算"手势稳住了"
SWITCH_VOTES_REF = 2              # 切换判定: 换词要连续领先这么多次才允许替换当前词(防来回跳)
                               # ↑ 这俩是"防抖轻重"的主旋钮: 调大=更稳但更慢, 调小=更快但可能跳字
RELEASE_NO_HAND_REF = 4           # 手离开画面连续几帧 = 动作做完了(清帧窗 + 解锁同一个词)。
                                  # 实测两头不讨好, 别乱动: 设2 -> MediaPipe漏检就清缓冲 -> 漏词;
                                  # 设8 -> 旧动作还留在缓冲里 -> 同一个词反复出(多词4涨到17)。4是甜点
RELEASE_NO_WORD_REF = 3           # 连续几次判不出有效词 = 解锁, 同一个词可以再出一次
MIN_EMIT_GAP_REF = 3              # 两次出词至少隔这么多帧。注意是帧不是秒 —— 用秒的话
                                  # CPU 一卡就把正常的下一个词当"间隔太短"吞掉(实测漏词翻倍)

SCORE_GATE = 1.0                  # 再严一点的闸: 距离必须 < 门槛×这个值才算认出来
LIVE_BUF_SEC = 2.0                # 帧缓冲保留最近几秒的动作

# ---- 下面是实际生效的值, 由 apply_fps() 按帧率自动换算, 别手动改 ----
LIVE_WINS = LIVE_WINS_REF
JUDGE_EVERY = JUDGE_EVERY_REF
JUDGE_WINDOW = JUDGE_WINDOW_REF
STABLE_VOTES = STABLE_VOTES_REF
SWITCH_VOTES = SWITCH_VOTES_REF
RELEASE_NO_HAND = RELEASE_NO_HAND_REF
RELEASE_NO_WORD = RELEASE_NO_WORD_REF
MIN_EMIT_GAP_FRAMES = MIN_EMIT_GAP_REF
LIVE_BUF = 60
_applied_fps = 0.0


def apply_fps(fps):
    """按摄像头实测帧率把所有"按帧计"的参数等比缩放。

    为什么必须做: 上面这些值原本是按 30fps 定的。摄像头只有 15fps 时,
    一个同样时长(比如1秒)的动作只有一半的帧, 还要求"滑窗5次判定""连续3票"的话,
    词还没判完人已经换下一个动作了。实测 15fps 下不做换算时整句正确率从 80%
    掉到 20%、漏词 9 个; 换算后回到 100%。
    写死帧数的话, 换个电脑/换个摄像头就会莫名其妙地吞词。
    """
    global LIVE_WINS, JUDGE_EVERY, JUDGE_WINDOW, STABLE_VOTES, SWITCH_VOTES
    global RELEASE_NO_HAND, RELEASE_NO_WORD, MIN_EMIT_GAP_FRAMES, LIVE_BUF, _applied_fps
    if fps <= 1 or abs(fps - _applied_fps) < 1.0:
        return False
    k = fps / FPS_REF
    LIVE_WINS = tuple(max(6, int(round(w * k))) for w in LIVE_WINS_REF)
    JUDGE_EVERY = 1 if fps < 20 else JUDGE_EVERY_REF
    JUDGE_WINDOW = max(3, int(round(JUDGE_WINDOW_REF * k)))
    STABLE_VOTES = max(2, min(int(round(STABLE_VOTES_REF * k)), JUDGE_WINDOW))
    SWITCH_VOTES = max(1, int(round(SWITCH_VOTES_REF * k)))
    RELEASE_NO_HAND = max(2, int(round(RELEASE_NO_HAND_REF * k)))
    RELEASE_NO_WORD = max(2, int(round(RELEASE_NO_WORD_REF * k)))
    MIN_EMIT_GAP_FRAMES = max(1, int(round(MIN_EMIT_GAP_REF * k)))
    LIVE_BUF = max(24, int(round(LIVE_BUF_SEC * fps)))
    _applied_fps = fps
    print(f"[自适应] 实测 {fps:.1f}fps -> 判词窗{LIVE_WINS} 每{JUDGE_EVERY}帧判一次 "
          f"滑窗{JUDGE_WINDOW}需{STABLE_VOTES}票 换词需{SWITCH_VOTES}次")
    return True

MERGE_PHRASES = True           # True=把相邻单字按词组表合并成完整词语(如 你+好 -> 你好)
WEAK_DROP = True               # True=丢掉"勉强达标"的可疑词(疑似误识别乱码)
WEAK_RATIO = 0.98              # 距离/门槛 >= 这个值才算"勉强"。实测正常词基本在 0.2~0.9
                               # (sign_log.txt 里能查到每次的分数), 所以 0.98 几乎不会误删;
                               #  想更狠就往 0.90 调, 想完全不删就 WEAK_DROP=False
# ====================================================================

_last_sent = []                # 上一次 Enter 发出去的词(z 还能改)


def _classify_rank(seg):
    """和 _classify_frames 同一套判定, 但把候选词都留着并按"相对分数"排好。
       为什么要留: 过渡动作常常同时像好几个词, 只看第一名的话,
       中间那几帧模糊的姿势会被硬认成某个词, 句子里就多出一个词。
       (试过"领先第二名才认", 真实数据上太严会让词漏掉, 已放弃。)"""
    best = None
    for pref in ("Right", "Left"):
        arr = _normalize_seq(seg, pref)
        if arr is None or len(arr) < 3:
            continue
        f = finalize(arr)
        one = not bool(np.any(arr[:, 63:]))
        d = np.linalg.norm(MX - f, axis=1)
        if one and TWO:
            allow = np.ones(len(My), dtype=bool)
            for _w, _ix in WORD_IDX.items():
                if TWO.get(_w, False):
                    allow[_ix] = False
            if not allow.any():
                continue
            d = np.where(allow, d, np.inf)
        top = np.argsort(d)[:K]
        votes = [str(My[j]) for j in top]
        ref_sc, ref_th = (TY1, TH1) if one else (TY, TH)
        cand = []
        for w in dict.fromkeys(votes):          # 保序去重
            ds = [d[j] for j in top if str(My[j]) == w]
            mdd = float(np.mean(ds))
            th = float(ref_th.get(w, DIST_THRESHOLD))
            cand.append((mdd / max(float(ref_sc.get(w, 4.0)), 0.8), w, mdd, th,
                         one, votes.count(w)))
        # 选词规则必须和线上一致: 先看票数(kNN 投票), 票数打平才比相对分数。
        # 单纯按分数排会让"票少但离得近"的词翻上来, 同一段动作里来回跳, 出词就断了。
        cand.sort(key=lambda c: (-c[5], c[0]))
        if cand and (best is None or cand[0][0] < best[0][0]):
            best = cand
    return best


def _live_classify(buf):
    """从最近的动作帧里用几种窗口长度各认一遍, 挑相对分数最好的那个。
       为什么多窗口: 人比划快慢能差一倍, 摄像头帧率也有高有低,
       固定窗口会在"动作比窗口长/短"时整段错位, 词就认不出来了。
       返回 (相对分数, 词, 距离, 门槛, 单手?, 领先第二名多少)"""
    best, fallback = None, None
    for W in LIVE_WINS:
        if len(buf) < W:
            continue
        cand = _classify_rank(buf[-W:])
        if not cand:
            continue
        sc, pred, dd, th, oh = cand[0][:5]
        if dd < th and (best is None or sc < best[0]):
            best = (sc, pred, dd, th, oh)
        if fallback is None or sc < fallback[0]:
            fallback = (sc, pred, dd, th, oh)
    return best if best is not None else fallback


# ---------------- 词组合并 / 纠错（纯后处理, 不参与模型） ----------------
# 校园高频词组表: 连着比出来的单字会按这张表贪心最长匹配, 拼成完整词。
# 想加校园专属词(楼名/课程名/老师称谓等)直接往这里加就行, 不用重训模型。
CAMPUS_PHRASES = [
    "西安培华学院", "培华学院", "培华",
    "学校", "食堂", "教室", "图书馆", "宿舍", "操场", "办公室", "教学楼", "实训楼",
    "辅导员", "老师", "同学", "班长", "校长",
    "请假", "上课", "下课", "考试", "报到", "注册", "毕业", "学号",
    "谢谢", "对不起", "没关系", "你好", "再见",
]
# 模型自带的词也算已知词: 保证"学校/名字/谢谢"这种本来就是词的不会被拆散
PHRASE_SET = set(CAMPUS_PHRASES) | set(str(w) for w in WORD_IDX)
_PHRASE_MAXLEN = max((len(p) for p in PHRASE_SET), default=1)


def polish_words(hits):
    """【后处理】把一串已确认的手势整理成最终上屏的词序列。
       1) 相邻重复去重: 手一直比着同一个姿势时, 同一个词不重复记
       2) 词组合并: 把相邻单字按词组表拼成完整词语(贪心最长匹配, 如 你+好 -> 你好)
       3) 弱词过滤: 丢掉刚好擦着门槛的"勉强词", 通常是过渡姿势造成的误识别
       hits: [[词, 置信度(距离/门槛)], ...]  -> 返回 ["词", ...]
    """
    if not hits:
        return []
    # --- 1. 相邻去重 ---
    seq = []
    for w, sc in hits:
        if seq and seq[-1][0] == w:
            seq[-1][1] = min(seq[-1][1], sc)     # 同一个词留下更稳的那次分数
            continue
        seq.append([w, sc])

    if not MERGE_PHRASES and not WEAK_DROP:
        return [w for w, _ in seq]

    # --- 2. 摊平成字流, 再贪心最长匹配 ---
    chars, owner = [], []
    for i, (w, sc) in enumerate(seq):
        for ch in w:
            chars.append(ch)
            owner.append(i)
    merged, i, n = [], 0, len(chars)
    while i < n:
        hit = None
        for L in range(min(_PHRASE_MAXLEN, n - i), 1, -1):
            if "".join(chars[i:i + L]) in PHRASE_SET:
                hit = L
                break
        if hit:
            # 合并后的置信度取参与合并的那些词里最稳的一个
            merged.append(["".join(chars[i:i + hit]),
                           min(seq[o][1] for o in owner[i:i + hit])])
            i += hit
        else:
            merged.append([chars[i], seq[owner[i]][1]])
            i += 1

    # --- 2b. 相邻"子串吸收": 一个动作的收尾帧常常被重复认成它的前半/后半
    #          (比如"你"还没收完就已经被认成"你好", 收完又出一次"好")
    #           -> 相邻两个词只要一个是另一个的子串, 就只留长的那个 ---
    if len(merged) > 1:
        keep = True
        while keep and len(merged) > 1:
            keep = False
            for k in range(len(merged) - 1):
                a, b = merged[k][0], merged[k + 1][0]
                if a and b and len(a) != len(b):
                    short = a if len(a) < len(b) else b
                    long_ = b if len(a) < len(b) else a
                    if short in long_:                       # 子串 -> 干掉短的
                        merged.pop(k if len(a) < len(b) else k + 1)
                        keep = True
                        break

    # --- 3. 弱词过滤(可选): 只对"单字 + 刚好过门槛"下手, 不影响成词的结果 ---
    out = []
    for tok, sc in merged:
        if WEAK_DROP and len(tok) == 1 and sc >= WEAK_RATIO:
            print(f"   [疑似乱码已过滤] {tok} (分数{sc:.2f} >= {WEAK_RATIO})")
            continue
        out.append(tok)
    return out


def _emit_word(word, dist, th, one_hand):
    """收到一个稳定手势 -> 记进本句 -> 合并纠错 -> 立刻上屏(屏幕上是逐步拼接的句子)"""
    score = dist / max(th, 1e-6)      # 相对置信度: 越小越稳(1.0=刚好卡在门槛上)
    with _word_lock:
        _word_buf.append([word, score])
        if len(_word_buf) > MAX_WORDS:
            _word_buf.pop(0)
        hits = [list(h) for h in _word_buf]
    line = polish_words(hits)
    print(f"🤟 [手语] {word}  (距离{dist:.1f}/门槛{th:.1f} 分数{score:.2f} "
          f"{'单手' if one_hand else '双手'})  | 本句: {' + '.join(line)}")
    _log(f"{word} d={dist:.1f} 门槛{th:.1f} 分数{score:.2f} "
         f"{'单手' if one_hand else '双手'} 本句={line}")
    publish_sign("".join(line)[:18])
    return True


def _clear_words(msg="下一句从零开始"):
    with _word_lock:
        _word_buf.clear()
    publish_sign("")
    print(f"[清空] {msg}")


def flush_sentence():
    """按 Enter: 把攒到的词组句 -> 发屏幕 -> 本句清空(词还在 _last_sent, 能 z 撤回)"""
    global _last_sent
    with _word_lock:
        hits = [list(h) for h in _word_buf]
    line = polish_words(hits)
    if not line:
        print("[Enter] 本句还没有词 (先比划, 等屏幕上出现词再按)")
        return
    sentence = sentence_api.compose(line)
    show = (sentence or "".join(line))[:18]
    publish_sign(show)
    print(f"📝 [手语-发送] {' + '.join(line)} -> {show}")
    _last_sent = list(line)
    with _word_lock:
        _word_buf.clear()


def undo_word():
    """按 z: 撤掉本句最后一个词; 本句空了就改上一句(重新组句发一次)"""
    global _last_sent
    with _word_lock:
        if _word_buf:
            gone = _word_buf.pop()[0]
            line = polish_words([list(h) for h in _word_buf])
            print(f"[撤销] 删掉「{gone}」 | 本句: {' + '.join(line) if line else '(空)'}")
            publish_sign("".join(line)[:18])
            return
        if _last_sent:
            gone = _last_sent.pop()
            sentence = sentence_api.compose(list(_last_sent))
            show = (sentence or "".join(_last_sent))[:18] if _last_sent else " "
            publish_sign(show)
            print(f"[撤销] 上句删掉「{gone}」 -> 屏幕: {show}")
            return
    print("[撤销] 没有可撤的词")


class LiveRecognizer:
    """【后处理】连续流式识别状态机: 滑窗 + 防抖 + 稳定判定 + 切换判定。
       完全不碰 MediaPipe、不碰 kNN 模型 —— 只消费 _live_classify 的结果做决策。
       feed(一帧) -> 该出词时返回 (词, 距离, 门槛, 单手?)

       三条规则(对应上面参数区的旋钮):
         1) 稳定判定 STABLE_VOTES: 最近 JUDGE_WINDOW 次判定里同一个词票数够多,
            才认为"这个手势稳住了"。单帧抖动 (=None 或别的词) 不会出词。
         2) 切换判定 SWITCH_VOTES: 已经出一个词之后就占着位(self.shown),
            新词必须连续领先 SWITCH_VOTES 次才能接管, 防止两个词在边界来回跳。
            迟滞: 平票时偏向已经显示的那个词。
         3) 解锁 RELEASE_*: 手离开画面(RELEASE_NO_HAND 帧)或连续判不出有效词
            (RELEASE_NO_WORD 次) = 上一个动作结束了, 同一个词允许再次出现。

       出词后不清空帧缓冲(清空会白等半秒), 让窗口自然滚过去,
       所以词与词之间几乎无缝 —— 连贯比划时可以不间断地出词。"""

    def __init__(self):
        self.buf = []            # 最近若干帧的手部关键点(滑动窗口的数据来源)
        self.side_hist = []      # 每帧"有没有右手/左手", 用于手位校验
        self.frame_no = 0
        self.miss = 0            # 连续没检测到手的帧数
        self.no_word = 0         # 连续判不出有效词的次数
        self.judges = []         # 最近 JUDGE_WINDOW 次判定的结果(None=没认出来)
        self.run_word = None     # 当前连续领先的词
        self.run_len = 0         # 它连续领先了多少次判定
        self.shown = None        # 当前占着位的词(没换手势就不重复出)
        self.last_emit_f = -999  # 上次出词在第几帧(控 MIN_EMIT_GAP_FRAMES)
        self.pred, self.dist, self.th, self.one = "...", None, None, False

    def reset(self, full=True):
        """动作断了: 清判定滑窗并解锁同一个词; full=True 连帧缓冲一起清"""
        self.judges = []
        self.run_word, self.run_len = None, 0
        self.no_word = 0
        self.shown = None
        if full:
            self.buf.clear()
            self.side_hist.clear()

    def _rule_ok(self, word):
        """手位闸门: 这个词登记了"右手/左手"时, 最近几帧必须真有那只手。
           没登记的标签一律放行 —— 老词库行为完全不变。"""
        rule = sign_guard.get_hand_rule(word)
        if rule == "none":
            return True
        recent = self.side_hist[-10:]
        if not recent:
            return False
        hit = sum(1 for r, l in recent if (r if rule == "right" else l))
        return hit >= len(recent) * 0.6

    def _vote(self):
        """滑窗多数表决 -> (得票最多的词, 票数)。平票时偏向当前已显示的词(迟滞)"""
        cnt = {}
        for w in self.judges:
            if w:
                cnt[w] = cnt.get(w, 0) + 1
        if not cnt:
            return None, 0
        top = max(cnt.values())
        best = [w for w, c in cnt.items() if c == top]
        if self.shown in best:            # 迟滞: 平票就保持原样, 文字不会乱跳
            return self.shown, top
        return best[0], top

    def feed(self, fh):
        # ---- 1) 没手的帧: 手势之间最自然的分界 ----
        if fh is None:
            self.miss += 1
            self.no_word += 1
            self.judges.append(None)
            if len(self.judges) > JUDGE_WINDOW:
                self.judges.pop(0)
            if self.miss >= RELEASE_NO_HAND:
                # 手连续几帧不在画面 = 这个动作做完了 -> 清窗并解锁。
                # 不清的话手放下再抬起, 旧动作的帧还留在缓冲里 -> "现在 现在"重复出词
                self.reset(full=True)
            elif self.no_word >= RELEASE_NO_WORD:
                self.shown = None
            return None

        # ---- 2) 有手: 入缓冲, 每 JUDGE_EVERY 帧判一次 ----
        self.miss = 0
        self.side_hist.append((fh[0] is not None, fh[1] is not None))
        if len(self.side_hist) > LIVE_BUF:
            self.side_hist.pop(0)
        self.buf.append(fh)
        if len(self.buf) > LIVE_BUF:
            self.buf.pop(0)
        self.frame_no += 1
        if len(self.buf) < LIVE_WINS[0]:
            return None                          # 帧还不够一个最短窗口
        if JUDGE_EVERY > 1 and self.frame_no % JUDGE_EVERY:
            return None                          # 隔帧判, 省算力

        # ---- 3) 模型层的结果 kNN 分类, 这里只拿来投票, 不做任何修改 ----
        r = _live_classify(self.buf)
        if r is None:
            self.judges.append(None)
            if len(self.judges) > JUDGE_WINDOW:
                self.judges.pop(0)
            return None
        sc, pred, dd, th, oh = r
        self.pred, self.dist, self.th, self.one = pred, dd, th, oh
        # 唯一的硬标准: 距离是否小于这个词自己的门槛(TH/TH1)。
        # 别再拿"距离/典型距离"当第二道闸 —— 那是选词用的相对尺度,
        # 散度大的词(如"现在")会被它误杀。
        ok = dd < th * SCORE_GATE
        # ↑ SCORE_GATE: 1.0=只按门槛; 调小(如0.9)=要求离得更近才认,
        #   能压掉过渡姿势蹭出来的词, 但调太小会把散度大的词压没(宁可别低于0.85)

        self.judges.append(pred if ok else None)
        if len(self.judges) > JUDGE_WINDOW:
            self.judges.pop(0)

        # ---- 4) 连续领先计数: 看是不是"稳住了", 而不是一帧抖出来的 ----
        if ok and pred == self.run_word:
            self.run_len += 1
        elif ok:
            self.run_word, self.run_len = pred, 1
        else:
            self.run_word, self.run_len = None, 0

        # ---- 5) 解锁判定 ----
        if not ok:
            self.no_word += 1
            if self.no_word >= RELEASE_NO_WORD:
                self.shown = None                # 中间出现"不像任何词" = 分界
        else:
            self.no_word = 0

        # ---- 6) 稳定判定(滑窗多数表决) ----
        cand, votes = self._vote()
        if cand is None or votes < STABLE_VOTES:
            return None
        if cand == self.shown:
            return None                          # 还是刚才那个词 -> 不重复出

        # ---- 7) 切换判定: 新词要连续领先够次数才允许接管 ----
        if self.run_word != cand or self.run_len < SWITCH_VOTES:
            return None
        if self.frame_no - self.last_emit_f < MIN_EMIT_GAP_FRAMES:
            return None
        # ---- 8) 手位闸门: 登记了右手/左手的词, 手不对就不出词 ----
        if not self._rule_ok(cand):
            return None
        self.shown = cand
        self.last_emit_f = self.frame_no
        return (cand, dd, th, oh)


def sign_language_thread():
    global _fps
    cap = cv2.VideoCapture(0)
    rec = LiveRecognizer()
    _fps_n, _fps_t0 = 0, time.time()

    while True:
        success, frame = cap.read()
        if not success:
            continue
        _fps_n += 1
        if time.time() - _fps_t0 >= 1.0:
            _fps = _fps_n / (time.time() - _fps_t0)
            _fps_n, _fps_t0 = 0, time.time()
            apply_fps(_fps)          # 帧率变了就把所有"按帧"的参数重新换算一遍
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        fh = frame_hands(result, rgb)      # 传 rgb: 启用人脸误检过滤

        if fh is not None:
            for hand_lm in result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        got = rec.feed(fh)
        if got is not None:
            _emit_word(*got)

        # ---------------- 画面: 通道A(手语) + 通道B(语音) 同时展示 ----------------
        # (手掌关键点连线在上面对原 frame 画完, 这里只是叠字, 不影响骨骼图)
        frame = draw_cn(frame, f"当前手势: {rec.pred}", (170, 28),
                        color=(0, 255, 120) if rec.pred != "..." else (190, 190, 190))
        if rec.dist is not None and rec.th:
            frame = draw_cn(frame, f"{rec.dist:.1f}/{rec.th:.1f}", (330, 62),
                            small_font, (160, 220, 255))
            frame = draw_cn(frame, "双手" if not rec.one else "单手", (430, 62),
                            small_font, (120, 255, 160) if not rec.one else (255, 200, 120))
        frame = draw_cn(frame, f"@{_fps:.0f}fps", (560, 28), small_font, (150, 150, 150))

        # 通道A: 手语
        line = buf_snapshot()
        if line:
            frame = draw_cn(frame, "通道A 手语: " + " ".join(line), (14, 96),
                            color=(255, 230, 120))
        else:
            frame = draw_cn(frame, "通道A 手语: (做个动作, 词会自动拼上来)", (14, 96),
                            small_font, (150, 150, 150))
        # 通道B: 语音(另一路 ESP32 麦 -> 百度识别, 与手语各行显示互不干扰)
        vtext, vat = get_voice_text()
        if vtext:
            frame = draw_cn(frame, "通道B 语音: " + vtext, (14, 136),
                            color=(120, 255, 200))
            if time.time() - vat > 8.0:
                set_voice_text("")     # 太久没说话就把旧句子淡出, 免得和新的混在一起
        else:
            frame = draw_cn(frame, "通道B 语音: (等待说话...)", (14, 136),
                            small_font, (150, 150, 150))
        frame = draw_cn(frame, "Enter=输出整句   z=撤最后一个词   c=清空重来   v=清语音   q=退出",
                        (14, 176), small_font, (150, 210, 255))

        cv2.imshow("WuShengYouYi (Press q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (13, 10):
            flush_sentence()
        elif key == ord("z"):
            undo_word()
        elif key == ord("c"):
            _clear_words()
        elif key == ord("v"):
            set_voice_text("")            # 清掉屏上那行语音文字(不影响手语)
            publish_audio_text("")
            print("[清空] 已清屏: 通道B 语音")
    cap.release()
    cv2.destroyAllWindows()

# ================= 功能 2：语音通道（百度 短语音识别, 异步不堵车） =================
import queue
_audio_q = queue.Queue(maxsize=1)   # 只留最新一块音频, 旧的丢掉防延迟堆积

# 通道B 的最新识别结果(摄像头窗口要把语音和手语同屏显示, 所以缓存一句)
_voice_text = ""
_voice_at = 0.0
_voice_lock = threading.Lock()

def get_voice_text():
    with _voice_lock:
        return _voice_text, _voice_at

def set_voice_text(text):
    global _voice_text, _voice_at
    with _voice_lock:
        _voice_text = text
        _voice_at = time.time() if text else 0.0

def on_audio_message(client, userdata, msg):
    """MQTT 回调里只做轻活(音量判断+入队), 识别交给后台线程,
       否则百度识别那 1~2 秒会把整个 MQTT 通道堵住, 屏幕越来越迟钝"""
    pcm = np.frombuffer(msg.payload, dtype=np.int16)
    if pcm.size == 0:
        return
    # 注意: 不能用原始 rms 当音量 —— 录音里有 -11000 的直流偏置,
    # 未处理时 98% 能量是直流, "音量"永远很大, 门槛形同虚设。
    # 这里用去直流后的交流音量(真实人声强度)。
    level = voice_api.ac_level(msg.payload)
    print(f"🎤 [语音] 收到{len(msg.payload)}字节 人声音量{level:.0f}")
    if level < 100:                   # 2026-11-15 换 INMP441 数字麦后重标定:
                                      # 老模拟麦底噪289/说话500~900, 门槛400;
                                      # 数字麦底噪10~50/说话(300Hz高通后)200~500, 门槛改100
        print("⚠️ [语音] 安静或只有底噪, 跳过")
        return
    try:                              # 清掉积压的旧包, 永远只处理最新
        while True:
            _audio_q.get_nowait()
    except queue.Empty:
        pass
    try:
        _audio_q.put_nowait(msg.payload)
    except queue.Full:
        pass

def _looks_fake(text):
    """百度对着噪声会硬编出"对对对对对"这类假话, 这里挡掉.
       注意: 只挡"同一个字连出4次以上"这种; 不能挡两字交替(你好好好/谢谢谢谢都是正常人话)"""
    core = text.strip("，。？！、,.?! ")
    if re.search(r"(.)\1{3,}", core):               # 同一个字连出 4 次以上, 如 "对对对对对"
        return True
    if len(core) >= 8 and len(set(core)) <= 2:      # 8字以上仍只有两个不同的字, 才算可疑
        return True
    return False

def publish_voice_text(text, cost=0.0):
    """识别出的文字 -> 去标点限长 -> 发屏幕 (ESP32 麦 和 电脑麦 共用这一条出口)"""
    if not text:
        print(f"⚠️ [语音] 没识别出文字 (耗时{cost:.1f}s)")
        return
    if _looks_fake(text):
        print(f"⚠️ [语音] 疑似噪声误识别, 丢弃: {text} (耗时{cost:.1f}s)")
        return
    # 屏幕只有 320 像素宽, 去掉标点并限长, 否则字会挤出屏幕
    screen_text = text
    for ch in "，。？！、,.?! ":
        screen_text = screen_text.replace(ch, "")
    if len(screen_text) > 12:
        screen_text = screen_text[:12]
    publish_audio_text(screen_text)
    set_voice_text(screen_text)          # 同步到摄像头窗口, 两个通道同屏显示
    print(f"✅ [语音] 识别: {text} (耗时{cost:.1f}s) -> 屏幕显示: {screen_text}")

def voice_worker():
    """后台识别线程: 拿最新音频 -> 百度识别 -> 发屏幕"""
    while True:
        pcm_bytes = _audio_q.get()
        t0 = time.time()
        text = voice_api.transcribe(pcm_bytes, rate=8000)
        publish_voice_text(text, time.time() - t0)

def voice_receive_thread():
    """收 ESP32 上传的音频: 订阅 + 后台网络循环已由 loop_start() 负责
       (以前这里 loop_forever 会和后台循环抢同一个 socket, 断线后两边一起卡死)"""
    global _AUDIO_CB
    _AUDIO_CB = on_audio_message
    client.on_message = on_audio_message
    try:
        client.subscribe(TOPIC_AUDIO_SEND)
    except Exception as e:
        print(f"[MQTT] 订阅音频主题失败: {e}")

# ================= 启动 =================
if __name__ == "__main__":
    print("无声有译 主程序启动！")
    print_word_table()      # 每词门槛/典型距离/单手还是双手, 判错了先看这张表
    print("1. 通道A 手语: 连续流式识别 —— 不需要打完停下等, 连贯比划即可; "
          "同一个手势稳住够了自动出词, 手势换了才更新")
    print(f"   (防抖参数: 滑窗{JUDGE_WINDOW}次判定/需{STABLE_VOTES}票, 换词需连续{SWITCH_VOTES}次, "
          f"每{JUDGE_EVERY}帧判一次; 词组合并={'开' if MERGE_PHRASES else '关'})")
    print("   Enter=输出整句 (z=撤最后一个词 c=清空手语 v=清语音 q=退出)")
    print(f"   (组句方式: {'DeepSeek 大模型' if sentence_api.API_KEY else '内置规则, 填 Key 可走大模型'})")
    if VOICE_SOURCE == "pc":
        import voice_pc
        print("2. 语音通道: 电脑/耳麦麦克风 16kHz -> 百度识别 (音质好, 推荐)")
        voice_pc.start(lambda t: publish_voice_text(t), device=VOICE_PC_DEVICE)
    else:
        print("2. 语音通道: ESP32 板载麦克风 8kHz -> 百度识别 (异步, 不卡屏)")
        threading.Thread(target=voice_worker, daemon=True).start()
    threading.Thread(target=voice_receive_thread, daemon=True).start()
    # 注: sentence_worker(停顿几秒自动组句上屏)在实时出词模式下不再启动,
    #     现在只有按 Enter 才会组句发送 —— 什么时候发由你决定。
    sign_language_thread()
