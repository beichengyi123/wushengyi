# -*- coding: utf-8 -*-
"""
=====================================================================
 无声有译 · 电脑端 GUI 桌面翻译软件 (gui_app.py)
---------------------------------------------------------------------
 定位: 把原来的命令行版 1.py 包装成一个双击就能用的窗口程序,
       识别逻辑一行不改 —— 直接复用 1.py 里的模型、归一化、
       kNN 判定、LiveRecognizer 防抖状态机、词组合并表。

 两条输出分支(对应升级需求):
   分支1  -> 局域网发送给 ESP32-D: 先把词/句子交给 Flask 后端,
             板子每 2 秒轮询后端拿内容显示(沿用原有链路, 屏幕逻辑不变)
   分支2  -> 提交本站后端智能体(SpringBoot + LangChain4j):
             POST {AGENT_URL}  {words:[...], image:"snapshots/xxx.jpg"}
             返回 {valid, text, tag} -> 在界面上展示润色后的句子

 本文件做的事:
   - 手语识别(通道A): 摄像头 -> 关键点 -> kNN -> 防抖出词 -> 智能体润色 -> 上屏
   - 语音识别(通道B): ESP32 板载 INMP441 录音 -> MQTT -> 百度转文字 -> 上屏
     (以前写在 1.py 的 __main__ 里, 打包成软件后不执行, 所以板子能录不能出字;
      现在搬进软件本体, 双击 exe 即可, 不需要开任何 Python 脚本)

 本文件不做的事(边界):
   - 不直连 DeepSeek(密钥只在后端, 软件只调自己的后端接口)
   - 不修改识别模型与判定参数

 运行:  python gui_app.py        (开发机: D:\\python\\python.exe gui_app.py)
 打包:  python build_exe.py      (生成 dist\\无声有译.exe, 免 Python 环境)
=====================================================================
"""

import os
import sys
import json
import time
import types
import queue
import base64
import shutil
import socket
import threading
import importlib.util
import urllib.request
import urllib.error
import urllib.parse
import subprocess
import webbrowser
import io
from datetime import datetime

# ---------------- 路径: 程序在哪都能找到项目根 ----------------
GUI_DIR = os.path.dirname(os.path.abspath(__file__))
# PyInstaller 打包后 __file__ 在临时解包目录, 用 exe 所在目录作为"程序目录"
APP_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else GUI_DIR
BASE_DIR = os.path.dirname(GUI_DIR) if not getattr(sys, "frozen", False) else APP_DIR

sys.path.insert(0, BASE_DIR)
# segno(离线二维码库)在 gui/_libs 下, 打包时由 build_exe 用 --paths 一并收集
_LIBS_DIR = os.path.join(GUI_DIR, "_libs")
if os.path.isdir(_LIBS_DIR) and _LIBS_DIR not in sys.path:
    sys.path.insert(0, _LIBS_DIR)

import cv2                     # noqa: E402
import numpy as np             # noqa: E402
import tkinter as tk           # noqa: E402
from tkinter import ttk, messagebox   # noqa: E402
from PIL import Image, ImageTk        # noqa: E402

# ---------------- 可改的配置(第一次运行会在 gui/config.json 生成) ----------------
DEFAULT_CFG = {
    "backend": "http://127.0.0.1:5000",          # Flask 后端(局域网内可写电脑IP)
    "agent": "http://127.0.0.1:8080/api/agent/compose",   # SpringBoot 智能体接口
    "camera": 0,                                  # 摄像头编号, 外接摄像头可改 1
    "push_polished_to_tft": True,                 # 智能体润色结果是否也推给 TFT 屏
    "snapshot": True,                             # 出词时是否自动抓拍存档
    # ---- 全自动成句: 手在比划, 不需要按任何按钮 ----
    "auto_finalize": True,                        # 停顿后自动成句/润色/上屏/入库
    "auto_pause_sec": 2.5,                        # 最后一个词之后静止多久算一句话说完
    "auto_agent": True,                           # 自动成句时是否调用智能体润色
    # ---- 语音通道(ESP32 INMP441 -> 百度短语音识别) ----
    "voice_enabled": True,                        # 启动软件就开语音, 说话自动上屏
    "voice_min_level": 100,                       # 人声音量门槛(去直流后交流RMS): 数字麦静音10~50, 说话200+
    # ---- 以下是"打开桌面端自动带起两个服务"用到的 ----
    "autostart_services": True,                   # 启动软件时自动拉起 后端+智能体
    "project_root": "",                           # 项目根目录, 留空=自动查找
    "python_exe": "",                             # 跑 Flask 的 python, 留空=自动查找
    "java_exe": "",                               # 跑智能体的 java, 留空=自动查找
}
CFG_PATH = os.path.join(APP_DIR, "config.json")


def load_cfg():
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(CFG_PATH):
        try:
            cfg.update(json.load(open(CFG_PATH, encoding="utf-8")))
        except Exception:
            pass
    else:
        try:
            json.dump(cfg, open(CFG_PATH, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
        except Exception:
            pass
    return cfg


CFG = load_cfg()

LOG_DIR = os.path.join(APP_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


# =====================================================================
# 零、自动带起两个后台服务(Flask 5000 + 智能体 8080)
#    双击桌面端 -> 软件自己把服务拉起来 -> 顶栏亮成"后端在线/智能体在线"
#    服务已经在跑就跳过(不会重复开, 也不会抢端口)
# =====================================================================
def _find_project_root():
    """定位"项目根": 必须同时有 web/app.py。exe 打包后被放在英文路径, 与项目分离, 所以要多处找。"""
    def ok(p):
        return bool(p) and os.path.isfile(os.path.join(p, "web", "app.py"))

    cands = []
    # 0) 打包时写下的路径提示(优先级最高, 由 build_exe.py 生成)
    for hint_dir in (APP_DIR, os.path.dirname(APP_DIR)):
        hint = os.path.join(hint_dir, "root_hint.txt")
        if os.path.isfile(hint):
            try:
                cands.append(open(hint, encoding="utf-8").read().strip())
            except Exception:
                pass
    # 1) 配置文件 / 环境变量
    if CFG.get("project_root"):
        cands.append(CFG["project_root"])
    if os.environ.get("WSY_ROOT"):
        cands.append(os.environ["WSY_ROOT"])
    # 2) 从程序目录逐级往上找
    d = APP_DIR
    for _ in range(6):
        cands.append(d)
        d = os.path.dirname(d)
    # 3) 桌面常见位置兜底
    home = os.path.expanduser("~")
    for desk in ("Desktop", os.path.join("OneDrive", "Desktop"), "桌面"):
        for sub in ("无声有译-项目", "无声有译"):
            cands.append(os.path.join(home, desk, sub))

    for p in cands:
        if ok(p):
            return os.path.abspath(p)
    return ""


def _port_open(port, host="127.0.0.1", timeout=0.6):
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def _find_python():
    if CFG.get("python_exe") and os.path.isfile(CFG["python_exe"]):
        return CFG["python_exe"]
    # 开发模式(python gui_app.py)下, 自己就是解释器
    if not getattr(sys, "frozen", False) and "python" in os.path.basename(sys.executable).lower():
        return sys.executable
    for p in (r"D:\python\python.exe", r"C:\Python310\python.exe",
              r"C:\Python311\python.exe", r"C:\Python312\python.exe",
              r"C:\Python39\python.exe"):
        if os.path.isfile(p):
            return p
    return shutil.which("python") or shutil.which("python3") or ""


def _find_java():
    if CFG.get("java_exe") and os.path.isfile(CFG["java_exe"]):
        return CFG["java_exe"]
    jh = os.environ.get("JAVA_HOME")
    cands = []
    if jh:
        cands.append(os.path.join(jh, "bin", "java.exe"))
    cands += [r"D:\dev\jdk17\bin\java.exe",
              r"C:\Program Files\Java\jdk-17\bin\java.exe",
              r"C:\Program Files\Java\jdk-21\bin\java.exe"]
    for p in cands:
        if os.path.isfile(p):
            return p
    return shutil.which("java") or ""


def _lan_ip():
    """本机在局域网里的 IP(手机访问网站要用)。拿不到就返回 127.0.0.1"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))      # 只选路, 不真的发包
        return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"
    finally:
        try:
            s.close()
        except Exception:
            pass


def _site_url(lan=False):
    """网站地址: 本机用配置里的 backend, 手机用 局域网IP:端口"""
    base = CFG.get("backend", "http://127.0.0.1:5000").rstrip("/")
    if not lan:
        return base
    u = urllib.parse.urlparse(base)
    port = u.port or 5000
    return "http://%s:%d" % (_lan_ip(), port)


def _spawn(cmd, cwd, log_name):
    """后台起一个子进程, 不弹黑窗, 输出写进 logs/xxx.log"""
    flags = 0x08000000 if os.name == "nt" else 0      # CREATE_NO_WINDOW
    log_path = os.path.join(LOG_DIR, log_name)
    f = open(log_path, "ab", buffering=0)
    try:
        p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL,
                             stdout=f, stderr=f, creationflags=flags)
    finally:
        f.close()      # 句柄已被子进程继承, 父进程可以关掉
    return p


# =====================================================================
# 一、把 1.py 当成"识别核心库"加载
#    注: 以前这里用一个假 paho 顶替真库(怕模块级连接卡住导入), 结果屏幕收不到
#    软件里的识别结果。1.py 已改成非阻塞连接 + 掉线重连, 现在用真 paho。
# =====================================================================
def load_core():
    """导入 1.py -> 得到模型、阈值表、识别状态机等一切现成的东西

    注意: 以前这里会把 paho 换成空壳(怕 1.py 模块级的连接死循环卡住导入),
    结果就是"软件里认出来的字根本发不到屏幕"。现在 1.py 改成非阻塞连接 +
    掉线自检重连, 用真 paho, 屏幕才能真正跟着软件更新。"""
    path = os.path.join(BASE_DIR, "1.py")
    spec = importlib.util.spec_from_file_location("signcore", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["signcore"] = mod
    spec.loader.exec_module(mod)      # 执行模块级代码: 加载模型/建阈值表(不启动摄像头)
    return mod


# =====================================================================
# 二、后端通信(只用标准库, 免装 requests, 打包更小)
# =====================================================================
def _post_json(url, payload, timeout=4):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8"))


def _post_multipart(url, fields, filename, file_bytes, timeout=8):
    """上传一张 jpg + 若干表单字段(手写 multipart, 不引入第三方库)"""
    boundary = "----wushengyouyi%s" % int(time.time() * 1000)
    body = b""
    for k, v in fields.items():
        body += ("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                 % (boundary, k, v)).encode("utf-8")
    body += ("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"
             "Content-Type: image/jpeg\r\n\r\n" % (boundary, filename)).encode("utf-8")
    body += file_bytes + b"\r\n"
    body += ("--%s--\r\n" % boundary).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "multipart/form-data; boundary=%s" % boundary})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8"))


def backend_alive():
    try:
        urllib.request.urlopen(CFG["backend"] + "/api/state", timeout=1.5).read()
        return True
    except Exception:
        return False


# =====================================================================
# 三、识别线程: 摄像头 -> 关键点 -> LiveRecognizer -> 出词
# =====================================================================
class CameraWorker(threading.Thread):
    """独立线程跑摄像头, 出词结果放进 self.events 队列, 由界面线程取走显示"""

    def __init__(self, core, events):
        super().__init__(daemon=True)
        self.core = core
        self.events = events          # queue.Queue: ("frame"|"word"|"status"|"error", ...)
        self.running = True
        self.cap = None
        self.fps = 0.0
        self.last_frame = None        # 给抓拍用(最新一帧原图)
        self.rec = core.LiveRecognizer()

    # ---- 摄像头开关 ----
    def open_camera(self):
        idx = CFG.get("camera", 0)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)   # DSHOW: Windows 上打开更快
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        return cap if cap.isOpened() else None

    def run(self):
        self.cap = self.open_camera()
        if self.cap is None:
            self.events.put(("error", "打不开摄像头: 检查是否被其它程序占用"))
            return
        self.events.put(("log", "摄像头线程已启动"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        n, t0 = 0, time.time()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            # 帧率统计: 每秒更新一次, 并把"按帧"的防抖参数按帧率重算(同 1.py)
            n += 1
            if time.time() - t0 >= 1.0:
                self.fps = n / (time.time() - t0)
                n, t0 = 0, time.time()
                try:
                    self.core.apply_fps(self.fps)
                except Exception:
                    pass

            frame = cv2.flip(frame, 1)                       # 镜像, 与实际动作一致
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.core.hands.process(rgb)
            fh = self.core.frame_hands(result, rgb)           # (右手, 左手) 或 None; 带人脸误检过滤

            # 画 21 点骨架(A 通道可视化)
            if fh is not None and result.multi_hand_landmarks:
                for lm in result.multi_hand_landmarks:
                    self.core.mp.solutions.drawing_utils.draw_landmarks(
                        frame, lm, self.core.mp_hands.HAND_CONNECTIONS)

            got = self.rec.feed(fh)
            if got is not None:
                word, dist, th, one_hand = got
                self.events.put(("word", word, dist, th, one_hand))

            self.last_frame = frame.copy()
            self.events.put(("frame", frame,
                             self.rec.pred, self.rec.dist, self.rec.th, self.rec.one,
                             self.fps))
        self.cap.release()
        self.events.put(("log", "摄像头线程已退出(正常停止)"))

    def stop(self):
        self.running = False


# =====================================================================
# 三·B、语音通道: ESP32(INMP441) 上传音频 -> 百度转文字 -> 屏幕/界面
#     原来这段只在 1.py 直接运行时才启动(__main__ 里), 软件把它当库加载,
#     所以打包后的 exe 里语音根本没人订阅 —— 板子显示"上传成功"却永远不出字。
#     现在把订阅和识别搬进软件本体, 双击 exe 就能用, 不需要开任何脚本。
# =====================================================================
class VoiceWorker(threading.Thread):
    """收 ESP32 的录音 -> 百度识别 -> 界面 + 屏幕

    两条接收通道, 先到先得:
     1) 局域网直连 (TCP 5001): 板子直连本机, 48KB 音频几十毫秒到, 是主通道
     2) 公网 MQTT: 老路子, 板子连不上本机时自动回退, 慢但能兜底
    """

    PC_INFO_TOPIC = "aiot/pc/info"           # 把本机局域网 IP 发到板上(保留消息)
    TCP_PORT = 5001

    def __init__(self, core, events):
        super().__init__(daemon=True)
        self.core = core
        self.events = events
        self.running = True
        self.q = queue.Queue(maxsize=1)      # 只留最新一段, 防延迟堆积
        self.enabled = bool(CFG.get("voice_enabled", True))
        self.attached = False
        self.api = None
        self.last_level = 0.0
        self.pc_ip = ""

    # ---- 加载百度接口(打包后 voice_api.py 在程序目录, 直接 import) ----
    def _load_api(self):
        try:
            import voice_api
            return voice_api
        except Exception as e:
            self.events.put(("log", "语音模块加载失败: %s" % e))
            return None

    # ---- 把音频回调挂到 1.py 的客户端上(不另开连接, 免得两个客户端互相踢) ----
    def attach(self):
        client = getattr(self.core, "client", None)
        if client is None:
            return False
        try:
            # 重连时 1.py 会重建客户端并用 _AUDIO_CB 重新挂回调, 所以要一起设
            self.core._AUDIO_CB = self._on_msg
            client.on_message = self._on_msg
            client.subscribe(self.core.TOPIC_AUDIO_SEND)
            self.attached = True
            self.events.put(("voice_state", "ready", 0))
            self._publish_pc_info(client)
            return True
        except Exception as e:
            self.events.put(("log", "订阅语音主题失败: %s" % e))
            return False

    # ---------------- 局域网直连(主通道) ----------------
    def _lan_ip(self):
        """取本机在局域网里的 IP(不会真的发包, 只是让系统选出口网卡)"""
        ip = ""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.3)
            s.connect(("223.5.5.5", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            ip = ""
        if not ip or ip.startswith("127."):
            try:
                ip = socket.gethostbyname(socket.gethostname())
            except Exception:
                ip = ""
        return ip

    def _publish_pc_info(self, client=None):
        """把本机 IP 用保留消息发给板子, 板子收到就改走局域网直连"""
        client = client or getattr(self.core, "client", None)
        if client is None:
            return
        ip = self._lan_ip()
        if not ip:
            return
        try:
            client.publish(self.PC_INFO_TOPIC, ip, retain=True)
            if ip != self.pc_ip:
                self.pc_ip = ip
                self.events.put(("log", "本机局域网地址已通知屏幕板: %s" % ip))
        except Exception:
            pass

    @staticmethod
    def _recv_exact(conn, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return b""
            buf.extend(chunk)
        return bytes(buf)

    def _tcp_loop(self):
        """监听 5001, 收板子直传的 4字节长度 + PCM16(8kHz)"""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("0.0.0.0", self.TCP_PORT))
            srv.listen(2)
            srv.settimeout(1.0)
        except Exception as e:
            self.events.put(("log", "局域网语音端口 %d 开不了(%s), 继续走 MQTT 老通道"
                             % (self.TCP_PORT, e)))
            try:
                srv.close()
            except Exception:
                pass
            return
        self.events.put(("log", "局域网语音通道已就绪: %s:%d"
                         % (self.pc_ip or "0.0.0.0", self.TCP_PORT)))
        while self.running:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except Exception:
                break
            conn.settimeout(10.0)
            try:
                while self.running:
                    hdr = self._recv_exact(conn, 4)
                    if not hdr:
                        break
                    n = int.from_bytes(hdr, "big")
                    if n <= 0 or n > 400000:      # 异常长度: 直接断开重来
                        break
                    pcm = self._recv_exact(conn, n)
                    if len(pcm) < n:
                        break
                    self._accept(pcm)
            except Exception:
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        try:
            srv.close()
        except Exception:
            pass

    # ---------------- 音频入口 ----------------
    def _on_msg(self, client, userdata, msg):
        """MQTT 回调: 只转交, 不在这里干活(会堵住整条 MQTT)"""
        self._accept(msg.payload)

    def _accept(self, pcm):
        """两条通道共用: 判音量 -> 入队。识别那 1~2 秒在别的线程做"""
        if not self.enabled or self.api is None:
            return
        try:
            level = float(self.api.ac_level(pcm))
        except Exception:
            return
        self.last_level = level
        if level < CFG.get("voice_min_level", 100):
            return
        self.events.put(("voice_state", "hearing", level))
        try:                              # 清掉积压的旧包, 永远只处理最新
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.q.put_nowait(pcm)
        except queue.Full:
            pass

    def run(self):
        self.api = self._load_api()
        if self.api is None:
            return
        threading.Thread(target=self._tcp_loop, daemon=True).start()
        for _ in range(20):               # 最多等 10 秒, 等 1.py 的 MQTT 连上
            if self.attach():
                break
            time.sleep(0.5)

        tick = 0
        while self.running:
            try:
                pcm = self.q.get(timeout=1.0)
            except queue.Empty:
                if not self.attached:     # 一直没连上: 每秒补一次订阅
                    self.attach()
                tick += 1
                if tick % 30 == 0:        # 每 30 秒重发一次 IP(IP 变了板子能跟上)
                    self._publish_pc_info()
                continue
            if not self.enabled:
                continue
            self.events.put(("voice_state", "busy", self.last_level))
            t0 = time.time()
            try:
                text = self.api.transcribe(pcm, rate=8000)
            except Exception as e:
                self.events.put(("log", "语音识别异常: %s" % e))
                continue
            self.events.put(("voice", text, time.time() - t0))

    def set_enabled(self, on):
        self.enabled = bool(on)
        if not on:
            try:
                while True:
                    self.q.get_nowait()
            except queue.Empty:
                pass

    def stop(self):
        self.running = False


# =====================================================================
# 四、主界面
# =====================================================================
class App:
    def __init__(self, root, core):
        self.root = root
        self.core = core
        self.events = queue.Queue()
        self.worker = None
        self.vworker = None      # 语音通道线程

        self.hits = []           # [[词, 分数], ...] 与 1.py 的 _word_buf 同结构
        self.last_sent = []      # 上一次发送的词, 供"撤销"后恢复
        self.agent_result = None
        self._auto_timer = None  # 停顿自动成句的定时器句柄
        self._ctx = []           # 最近几句成品句, 作为智能体上下文
        self.backend_ok = False
        self.snap_dir = os.path.join(APP_DIR, "snapshots")
        os.makedirs(self.snap_dir, exist_ok=True)
        self.procs = {}          # 本程序拉起的服务进程 {名字: Popen}
        self.booting = False     # 刚开始自动拉服务时, 别让探活把状态刷成"离线"

        root.title("无声有译 · 手语翻译桌面端")
        root.geometry("1180x760")
        root.minsize(1080, 700)
        root.configure(bg="#1b1d22")
        try:
            ico = os.path.join(APP_DIR, "icon.ico")
            if os.path.exists(ico):
                root.iconbitmap(default=ico)     # 窗口左上角/任务栏图标
        except Exception:
            pass

        self._build_ui()
        self.start_camera()
        self.start_voice()
        self.start_collect_watchdog()
        self.root.after(30, self._pump)
        self.root.after(500, self._autostart_services)
        self.root.after(9000, self._check_services)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- 界面骨架 ----------------
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # 顶部标题条
        top = tk.Frame(self.root, bg="#24272e", height=52)
        top.pack(side="top", fill="x")
        tk.Label(top, text="无声有译 · 手语翻译桌面端", bg="#24272e", fg="#eaeaea",
                 font=("Microsoft YaHei UI", 15, "bold")).pack(side="left", padx=16, pady=10)
        self.lbl_be = tk.Label(top, text="● 后端离线", bg="#24272e", fg="#c46b6b",
                               font=("Microsoft YaHei UI", 10))
        self.lbl_be.pack(side="right", padx=8)
        self.lbl_agent = tk.Label(top, text="● 智能体未知", bg="#24272e", fg="#8a8a8a",
                                  font=("Microsoft YaHei UI", 10))
        self.lbl_agent.pack(side="right", padx=8)

        body = tk.Frame(self.root, bg="#1b1d22")
        body.pack(fill="both", expand=True, padx=12, pady=10)

        # 左: 视频预览
        left = tk.Frame(body, bg="#1b1d22")
        left.pack(side="left", fill="both", expand=False)
        self.video = tk.Label(left, bg="#000000", width=720, height=540)
        self.video.pack()
        self.lbl_hint = tk.Label(left, text="正在打开摄像头…", bg="#1b1d22", fg="#9aa0a6",
                                 font=("Microsoft YaHei UI", 10))
        self.lbl_hint.pack(pady=6)

        # 右: 信息面板
        right = tk.Frame(body, bg="#1b1d22")
        right.pack(side="left", fill="both", expand=True, padx=(14, 0))

        # 实时状态
        card1 = self._card(right, "实时状态")
        self.lbl_gesture = tk.Label(card1, text="当前手势: —", bg="#24272e", fg="#5ddc9a",
                                     font=("Microsoft YaHei UI", 13, "bold"), anchor="w")
        self.lbl_gesture.pack(fill="x", padx=12, pady=(2, 2))
        self.lbl_metric = tk.Label(card1, text="距离/门槛: —   帧率: —", bg="#24272e",
                                    fg="#8c93a0", font=("Microsoft YaHei UI", 10), anchor="w")
        self.lbl_metric.pack(fill="x", padx=12, pady=(0, 10))

        # 词汇序列
        card2 = self._card(right, "识别出的词汇序列")
        self.lbl_words = tk.Label(card2, text="（比一个手势，词会自动拼上来）", bg="#24272e",
                                   fg="#f0d264", font=("Microsoft YaHei UI", 17, "bold"),
                                   wraplength=360, justify="left", anchor="w")
        self.lbl_words.pack(fill="x", padx=12, pady=(2, 10))

        # 智能体结果
        card3 = self._card(right, "智能体润色结果（后端 LangChain4j）")
        self.lbl_agent_text = tk.Label(card3, text="（停手 2.5 秒自动成句，结果显示在这里）",
                                        bg="#24272e", fg="#cfd3da",
                                        font=("Microsoft YaHei UI", 14), wraplength=360,
                                        justify="left", anchor="w")
        self.lbl_agent_text.pack(fill="x", padx=12, pady=(2, 4))
        self.lbl_agent_tag = tk.Label(card3, text="", bg="#24272e", fg="#7fb2ff",
                                       font=("Microsoft YaHei UI", 10), anchor="w")
        self.lbl_agent_tag.pack(fill="x", padx=12, pady=(0, 10))

        # 语音通道(通道B): ESP32 麦克风说话 -> 文字
        card_v = self._card(right, "语音通道（通道B · 板载麦克风说话 → 自动转文字上屏）")
        self.lbl_voice = tk.Label(card_v, text="（对着麦克风正常说话，文字自动出在这里）",
                                   bg="#24272e", fg="#8c93a0",
                                   font=("Microsoft YaHei UI", 14, "bold"), wraplength=360,
                                   justify="left", anchor="w")
        self.lbl_voice.pack(fill="x", padx=12, pady=(2, 2))
        self.lbl_voice_meta = tk.Label(card_v, text="● 正在连接语音通道…", bg="#24272e",
                                        fg="#7d8493", font=("Microsoft YaHei UI", 9), anchor="w")
        self.lbl_voice_meta.pack(fill="x", padx=12, pady=(0, 8))

        # 按钮区
        btns = tk.Frame(right, bg="#1b1d22")
        btns.pack(fill="x", pady=(4, 8))
        self._btn(btns, "▶ 立即上屏(手动)", self.on_send, "#2f7d4f").grid(row=0, column=0, sticky="ew", padx=3, pady=3)
        self._btn(btns, "↩ 撤销一个词", self.on_undo, "#3a3f4a").grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        self._btn(btns, "✕ 清空", self.on_clear, "#3a3f4a").grid(row=1, column=0, sticky="ew", padx=3, pady=3)
        self._btn(btns, "🤖 重新润色(手动)", self.on_agent, "#2b5f9e").grid(row=1, column=1, sticky="ew", padx=3, pady=3)
        # 网站入口: 不用再手动复制地址
        self._btn(btns, "🌐 打开网站", self.on_open_site, "#2f5f7d").grid(row=2, column=0, sticky="ew", padx=3, pady=3)
        self._btn(btns, "📱 手机扫码", self.on_copy_lan, "#3a3f4a").grid(row=2, column=1, sticky="ew", padx=3, pady=3)
        # 语音通道开关: 说话自动转文字; 不想让它收音时点一下关掉
        self.btn_voice = self._btn(btns, "", self.on_toggle_voice, "#2f7d4f")
        self.btn_voice.grid(row=3, column=0, columnspan=2, sticky="ew", padx=3, pady=3)
        self._sync_voice_btn()
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)

        # 日志
        card4 = self._card(right, "运行日志")
        self.log = tk.Text(card4, height=5, bg="#15171b", fg="#9fb0c0", bd=0,
                           font=("Consolas", 9), wrap="word")
        self.log.pack(fill="both", expand=True, padx=10, pady=(2, 10))

    def _card(self, parent, title):
        wrap = tk.Frame(parent, bg="#24272e")
        wrap.pack(fill="x", pady=(0, 10))
        tk.Label(wrap, text=title, bg="#24272e", fg="#7d8493",
                 font=("Microsoft YaHei UI", 9)).pack(anchor="w", padx=12, pady=(8, 2))
        return wrap

    def _btn(self, parent, text, cmd, color):
        b = tk.Button(parent, text=text, command=cmd, bg=color, fg="white",
                      activebackground=color, activeforeground="white",
                      relief="flat", bd=0, font=("Microsoft YaHei UI", 11),
                      cursor="hand2", pady=9)
        return b

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.insert("end", "[%s] %s\n" % (ts, msg))
        self.log.see("end")

    # ---------------- 摄像头 ----------------
    def start_camera(self):
        self.worker = CameraWorker(self.core, self.events)
        self.worker.start()
        self.lbl_hint.config(text="摄像头已启动 · 比完手势停顿一下，自动成句并上屏，不用按键")

    # ---------------- 语音通道 ----------------
    def start_voice(self):
        """启动语音线程: 订阅 ESP32 上传的音频 -> 百度识别 -> 界面 + 屏幕"""
        self.vworker = VoiceWorker(self.core, self.events)
        self.vworker.start()
        self._log("语音通道已启动: ESP32 麦克风 → 百度识别 → 自动上屏")

    def _sync_voice_btn(self):
        on = bool(CFG.get("voice_enabled", True))
        if not hasattr(self, "btn_voice"):
            return
        self.btn_voice.config(text=("🎤 语音通道：已开启（点击关闭）" if on
                                    else "🎤 语音通道：已关闭（点击开启）"),
                              bg=("#2f7d4f" if on else "#3a3f4a"),
                              activebackground=("#2f7d4f" if on else "#3a3f4a"))

    def on_toggle_voice(self):
        on = not bool(CFG.get("voice_enabled", True))
        CFG["voice_enabled"] = on
        try:
            json.dump(CFG, open(CFG_PATH, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)     # 记住选择, 下次开软件照旧
        except Exception:
            pass
        if self.vworker is not None:
            self.vworker.set_enabled(on)
        self._sync_voice_btn()
        self.lbl_voice_meta.config(text="● 语音通道已开启，等说话…" if on else "● 语音通道已关闭",
                                    fg="#7d8493")
        self._log("语音通道: %s" % ("开启" if on else "关闭"))

    def _on_voice(self, text, cost):
        """语音识别结果: 界面显示 + 推 TFT 屏 + 记进网页历史"""
        text = (text or "").strip()
        if not text:
            self.lbl_voice_meta.config(text="● 没听清（耗时 %.1fs），再说一次" % cost, fg="#c9a06b")
            return

        # 百度对着噪声会硬编"对对对对对", 用 1.py 现成的判据挡一下
        fake = getattr(self.core, "_looks_fake", None)
        if fake is not None and fake(text):
            self.lbl_voice_meta.config(text="● 疑似噪声误识别，已丢弃（%.1fs）" % cost, fg="#c9a06b")
            self._log("语音疑似噪声误识别，已丢弃: %s" % text)
            return

        # 屏幕只有 320 像素宽: 去标点 + 限长, 否则字挤出屏幕(同 1.py)
        show = text
        for ch in "，。？！、,.?! ":
            show = show.replace(ch, "")
        if len(show) > 12:
            show = show[:12]

        self.lbl_voice.config(text=text, fg="#7fd7ff")
        self.lbl_voice_meta.config(text="● 已上屏：%s   （识别耗时 %.1fs）" % (show, cost), fg="#5ddc9a")
        self._log("语音: %s" % text)

        def work():
            # 屏幕: 走语音专属主题(和手语分开, 免得互相覆盖)
            try:
                if hasattr(self.core, "publish_audio_text"):
                    if not self.core.publish_audio_text(show):
                        self._log("语音文字没送到屏幕(网络/MQTT 断了)，下一条会自动重试")
            except Exception as e:
                self._log("语音上屏异常: %s" % e)
            # 网页看板/历史记录
            try:
                _post_json(CFG["backend"] + "/api/display", {
                    "source": "voice", "event": "voice", "text": text,
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            except Exception:
                pass
        threading.Thread(target=work, daemon=True).start()

    # ---------------- 采集页让位看门狗 ----------------
    def start_collect_watchdog(self):
        """盯着网站的采集页: 页面一打开就把摄像头让给它, 页面关了自动拿回来。
           用户双手比划时再也不用手动关软件开网站了"""
        self._cam_yielded = False

        def watch():
            import urllib.request
            while True:
                time.sleep(1.5)
                try:
                    req = urllib.request.Request(CFG["backend"] + "/api/collect/busy")
                    with urllib.request.urlopen(req, timeout=3) as r:
                        import json as _json
                        busy = _json.loads(r.read().decode()).get("busy", False)
                except Exception:
                    continue          # 后端没起来/网络抖动: 不动摄像头
                if busy != self._cam_yielded:
                    self._cam_yielded = busy
                    self.events.put(("cam_yield", busy))

        threading.Thread(target=watch, daemon=True).start()

    def _on_cam_yield(self, busy):
        try:   # 摄像头归属变化写文件日志, 方便排查"让没让/收没收回"
            with open(os.path.join(CFG.get("project_root", "."), "logs", "cam_switch.log"),
                      "a", encoding="utf-8") as f:
                f.write("%s %s\n" % (datetime.now().strftime("%m-%d %H:%M:%S"),
                                     "YIELD 让给采集页" if busy else "RESUME 收回摄像头"))
        except Exception:
            pass
        if busy:
            if self.worker is not None:
                self.worker.stop()
                self.worker = None
            self.video.config(image="")
            self.video.config(text="摄像头已让给网页采集页\n\n关掉采集页后几秒自动恢复", fg="#fbbf24")
            self.lbl_hint.config(text="网页采集页已打开 · 摄像头已让给采集页（关掉采集页几秒后自动恢复）",
                                 fg="#fbbf24")
            self._log("采集页在线，摄像头已让出")
        else:
            self.video.config(text="", fg="#cfd3da")
            self.lbl_hint.config(fg="#cfd3da")
            self.start_camera()
            self._log("采集页已关闭，摄像头已恢复")

    # ---------------- 事件泵: 每 30ms 把线程结果搬到界面 ----------------
    def _pump(self):
        try:
            for _ in range(8):
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == "frame":
                    self._show_frame(ev[1])
                    self.lbl_gesture.config(text="当前手势: %s" % ev[2])
                    dist, th = ev[3], ev[4]
                    metric = "距离/门槛: %s" % ("—" if dist is None else "%.1f / %.1f" % (dist, th or 0))
                    self.lbl_metric.config(text="%s   %s   帧率: %.0f" %
                                                (metric, "双手" if dist is not None and not ev[5] else "单手",
                                                 ev[6]))
                elif kind == "word":
                    self._on_word(ev[1], ev[2], ev[3], ev[4])
                elif kind == "log":
                    self._log(ev[1])
                elif kind == "svc":
                    # (svc, 哪一个, 文字, 颜色)
                    lbl = self.lbl_be if ev[1] == "be" else self.lbl_agent
                    lbl.config(text=ev[2], fg=ev[3])
                elif kind == "agent_ui":
                    # (agent_ui, 润色后的句子, 标签) —— 由自动成句线程投递, 主线程更新控件
                    text, tag = ev[1], ev[2]
                    self.lbl_agent_text.config(text=text or "（空）", fg="#9fe1a0")
                    self.lbl_agent_tag.config(text=("标签: %s" % tag) if tag else "")
                elif kind == "error":
                    self._log("错误: " + ev[1])
                    self.lbl_hint.config(text=ev[1], fg="#e07b7b")
                elif kind == "voice":
                    # (voice, 识别文字, 耗时) —— 语音线程转写完成
                    self._on_voice(ev[1], ev[2])
                elif kind == "voice_state":
                    # (voice_state, ready|hearing|busy, 音量)
                    st, lv = ev[1], ev[2]
                    if st == "ready":
                        if self.lbl_voice.cget("fg") == "#8c93a0":
                            self.lbl_voice_meta.config(text="● 语音通道已就绪，对着麦克风说话",
                                                        fg="#5ddc9a")
                    elif st == "hearing":
                        self.lbl_voice_meta.config(text="● 听到声音（音量 %.0f），正在识别…" % lv,
                                                    fg="#7fd7ff")
                    elif st == "busy":
                        self.lbl_voice_meta.config(text="● 正在识别（音量 %.0f）…" % lv, fg="#7fd7ff")
                elif kind == "cam_yield":
                    # (cam_yield, True/False) —— 采集页开/关, 让出或恢复摄像头
                    self._on_cam_yield(ev[1])
        except queue.Empty:
            pass
        self.root.after(30, self._pump)

    def _show_frame(self, bgr):
        if getattr(self, "_cam_yielded", False):
            return      # 摄像头已让给采集页: 摄像头线程退出前塞的尾巴帧一律丢弃,
                        # 别把"已让出"的提示画面又盖成定格图像
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((720, 540), Image.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(img)
        self.video.config(image=self._tk_img)

    # ---------------- 出词 ----------------
    def _on_word(self, word, dist, th, one_hand):
        score = dist / max(th, 1e-6)
        self.hits.append([word, score])
        line = self.core.polish_words([list(h) for h in self.hits])
        self.lbl_words.config(text="  ·  ".join(line) if line else "（无）")
        self._log("手语: %s  距离%.1f/门槛%.1f 分数%.2f %s" %
                  (word, dist, th, score, "单手" if one_hand else "双手"))

        # 抓拍 + 上报(放后台做, 不卡识别循环)
        frame = None if self.worker is None else self.worker.last_frame
        threading.Thread(target=self._report_word,
                         args=(word, dist, th, one_hand, score, "".join(line),
                               None if frame is None else frame.copy()),
                         daemon=True).start()

        # 全自动: 每出一个词就重置"停顿计时", 手停下来就自动成句+润色+上屏
        self._arm_auto_finalize()

    # ---------------- 全自动成句(不需要按按钮) ----------------
    def _arm_auto_finalize(self):
        """每来一个新词就把定时器往后推; 停手 auto_pause_sec 秒后自动收句"""
        if not CFG.get("auto_finalize", True):
            return
        if self._auto_timer is not None:
            try:
                self.root.after_cancel(self._auto_timer)
            except Exception:
                pass
            self._auto_timer = None
        try:
            pause = float(CFG.get("auto_pause_sec", 2.5))
        except Exception:
            pause = 2.5
        self._auto_timer = self.root.after(int(max(1.0, pause) * 1000), self._auto_finalize)

    def _auto_finalize(self):
        """停顿到点: 把当前词串当成一句, 自动润色并推屏(手不用离开画面)"""
        self._auto_timer = None
        if not CFG.get("auto_finalize", True):
            return
        if not self.hits:
            return
        self._finalize(auto=True)

    def _finalize(self, auto=False):
        """成句统一出口: 原词串 -> (可选)智能体润色 -> 上屏 -> 落库 -> 清空缓冲"""
        words = [h[0] for h in self.hits]
        raw = "".join(self.core.polish_words([list(h) for h in self.hits]))
        if not raw:
            return
        self.last_sent = [list(h) for h in self.hits]
        self.hits = []
        self.lbl_words.config(text="（本句已成: %s）" % raw)
        tag_hint = "自动" if auto else "手动"
        self._log("[%s成句] %s" % (tag_hint, raw))

        def work():
            text, tag, short = raw, "", ""
            use_agent = CFG.get("auto_agent", True) and bool(CFG.get("agent"))
            if use_agent:
                payload = {"words": words, "raw": raw,
                           "context": self._ctx[-3:],
                           "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                try:
                    r = _post_json(CFG["agent"], payload, timeout=30)
                    if r.get("valid", r.get("ok", False)):
                        text = r.get("text") or r.get("sentence") or raw
                        tag = r.get("tag") or r.get("label") or ""
                        short = r.get("short") or ""
                    else:
                        self._log("智能体判定无效, 原词串上屏: %s" % raw)
                except Exception as e:
                    self._log("智能体不可达, 原词串上屏(%s)" % e)

            # 界面显示
            self.events.put(("agent_ui", text, tag))
            # 推屏幕: 有短句用短句(TFT 一行放得下), 没有就用整句
            show = short or text
            # 真的发 MQTT 给 TFT(以前这里只发网页后端, 屏幕收不到)
            self._tft(show)
            try:
                _post_json(CFG["backend"] + "/api/display", {
                    "source": "sign", "event": "final", "text": show,
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            except Exception as e:
                self._log("网页看板同步失败: %s" % e)
            # 落库由智能体服务自己回调 /api/agent_record 完成(避免一条句子存两遍),
            # 这里只在降级(智能体不可用)时补存原始词串, 保证历史里不丢这一句
            if text == raw:
                try:
                    _post_json(CFG["backend"] + "/api/agent_record", {
                        "words": words, "text": text, "tag": "",
                        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                except Exception as e:
                    self._log("整句入库失败: %s" % e)
            self._ctx.append(text)
            self._log("[上屏] %s%s" % (show, ("  [%s]" % tag) if tag else ""))
        threading.Thread(target=work, daemon=True).start()

    def _report_word(self, word, dist, th, one_hand, score, sentence, frame):
        """分支1: 词汇/分数入后端, 由板子轮询上屏; 同时抓拍存档"""
        image_path = ""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fname = "%s_%s.jpg" % (stamp, word)
        local = os.path.join(self.snap_dir, fname)

        if frame is not None and CFG.get("snapshot", True):
            try:
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok:
                    data = buf.tobytes()
                    open(local, "wb").write(data)          # 本地留一份, 断网也不丢
                    # 上传到网站(抓拍图跟随记录一起存)
                    try:
                        r = _post_multipart(CFG["backend"] + "/api/snapshot",
                                            {"word": word, "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                                            fname, data)
                        image_path = r.get("path", "")
                    except Exception as e:
                        self._log("抓拍上传失败(已存本地): %s" % e)
            except Exception as e:
                self._log("抓拍失败: %s" % e)

        try:
            _post_json(CFG["backend"] + "/api/sign_word", {
                "word": word, "dist": round(dist, 2), "th": round(th, 2),
                "score": round(score, 2), "hand": "单手" if one_hand else "双手",
                "sentence": sentence, "image": image_path,
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            _post_json(CFG["backend"] + "/api/display", {
                "source": "sign", "event": "word", "text": sentence,
                "score": round(score, 2),
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            if not self.backend_ok:
                self._log("后端已连接: " + CFG["backend"])
            self.backend_ok = True
        except Exception as e:
            self.backend_ok = False
            self._log("上报后端失败(检查后端是否启动): %s" % e)

        # 逐词拼接也实时上屏(和 1.py 独立运行时的观感一致), 停手后由润色结果覆盖
        self._tft(sentence)

    # ---------------- 按钮动作 ----------------
    def _current_sentence(self):
        return "".join(self.core.polish_words([list(h) for h in self.hits]))

    def _tft(self, text):
        """把文字发到 TFT 屏(真 MQTT, 1.py 的高层出口带重连重试)。
           放后台线程调用, 不阻塞界面和识别"""
        core = getattr(self, "core", None)
        if core is None or not hasattr(core, "publish_sign"):
            return

        def run():
            try:
                if not core.publish_sign(text or ""):
                    self._log("屏幕没收到(网络/MQTT 断了), 会自动重试下一条")
            except Exception as e:
                self._log("发送屏幕异常: %s" % e)

        threading.Thread(target=run, daemon=True).start()

    def on_send(self):
        """把当前这句整句发给屏幕(相当于 1.py 里按 Enter) —— 手动兜底, 平时全自动"""
        if self._auto_timer is not None:
            try:
                self.root.after_cancel(self._auto_timer)
            except Exception:
                pass
            self._auto_timer = None
        text = self._current_sentence()
        if not text:
            self._log("没有可发送的词")
            return
        self.last_sent = [list(h) for h in self.hits]
        self.hits = []
        self.lbl_words.config(text="（已发送，等下一个动作）")
        self.lbl_agent_text.config(text="（等智能体润色…）", fg="#cfd3da")
        self.lbl_agent_tag.config(text="")

        def work():
            self._tft(text)
            try:
                _post_json(CFG["backend"] + "/api/display", {
                    "source": "sign", "event": "final", "text": text,
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                self._log("已发送到屏幕: %s" % text)
            except Exception as e:
                self._log("发送失败: %s" % e)
        threading.Thread(target=work, daemon=True).start()

    def on_undo(self):
        if self.hits:
            w = self.hits.pop()
            self._log("撤销: %s" % w[0])
        elif self.last_sent:
            self.hits = self.last_sent
            self.last_sent = []
            self._log("恢复上一句的字词")
        line = self.core.polish_words([list(h) for h in self.hits])
        self.lbl_words.config(text="  ·  ".join(line) if line else "（无）")
        # 屏幕上也要跟着改, 否则用户看到的还是错的那句
        self._tft("".join(line))

    def on_clear(self):
        self.hits, self.last_sent = [], []
        self.lbl_words.config(text="（已清空）")
        self.lbl_agent_text.config(text="（停手 2.5 秒自动成句，结果显示在这里）", fg="#cfd3da")
        self.lbl_agent_tag.config(text="")
        self._tft("")

        def work():
            try:
                _post_json(CFG["backend"] + "/api/display", {"source": "sign", "event": "clear", "text": ""})
            except Exception:
                pass
        threading.Thread(target=work, daemon=True).start()
        self._log("已清空, 屏幕同步清空")

    # ---------------- 网站入口(不用手打地址) ----------------
    def on_open_site(self):
        """用默认浏览器打开后端网站: 本机 127.0.0.1:5000"""
        url = _site_url()
        if not backend_alive():
            self._log("后端还没就绪, 打不开网站; 等右上角亮绿灯再点(或已在启动中)")
            messagebox.showinfo(
                "网站还没起来",
                "后端服务还没就绪。\n\n"
                "· 右上角显示「后端启动中」请等 3~5 秒再点\n"
                "· 显示「后端离线」请看底下的运行日志\n\n"
                "地址是: %s" % url)
            return
        try:
            webbrowser.open(url)
            self._log("已用浏览器打开网站: %s" % url)
        except Exception as e:
            self._log("打开浏览器失败: %s (手动访问 %s)" % (e, url))

    def on_copy_lan(self):
        """弹出二维码: 手机相机一扫直达网站, 不用传地址"""
        import segno
        url = _site_url(lan=True)
        buf = io.BytesIO()
        segno.make(url, error="m").save(buf, kind="png", scale=8, border=2,
                                        dark="#111111", light="#ffffff")
        qr_img = Image.open(buf)

        win = tk.Toplevel(self.root)
        win.title("手机扫码打开网站")
        win.configure(bg="#ffffff")
        win.resizable(False, False)
        win.transient(self.root)
        self._qr_photo = ImageTk.PhotoImage(qr_img)   # 防止被垃圾回收
        tk.Label(win, image=self._qr_photo, bg="#ffffff").pack(padx=24, pady=(18, 8))
        tk.Label(win, text="手机相机对准扫码, 直达网站", bg="#ffffff",
                 fg="#333333", font=("Microsoft YaHei", 11)).pack(pady=(0, 2))
        tk.Label(win, text=url, bg="#ffffff", fg="#666666",
                 font=("Consolas", 10)).pack(pady=(0, 10))

        def copy_url():
            self.root.clipboard_clear()
            self.root.clipboard_append(url)
            self.root.update()
            self._log("已复制手机地址: %s" % url)

        tk.Button(win, text="复制地址(备用)", command=copy_url,
                  font=("Microsoft YaHei", 9), bg="#eef2f7", relief="flat",
                  padx=12, pady=4).pack(pady=(0, 16))
        win.bind("<Escape>", lambda e: win.destroy())
        self._log("已弹出手机扫码窗口: %s" % url)

    def on_agent(self):
        """分支2: 把当前词序列提交后端智能体, 拿润色句子回来"""
        line = self.core.polish_words([list(h) for h in self.hits]) or \
               self.core.polish_words([list(h) for h in self.last_sent])
        if not line:
            self._log("没有词汇可提交")
            return
        self.lbl_agent_text.config(text="智能体处理中…", fg="#8c93a0")
        self.lbl_agent_tag.config(text="")

        def work():
            payload = {"words": line,
                       "raw": " ".join(line),
                       "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            try:
                r = _post_json(CFG["agent"], payload, timeout=30)
            except Exception as e:
                self.lbl_agent_text.config(text="智能体服务未启动或不可达", fg="#e07b7b")
                self.lbl_agent_tag.config(text="%s" % CFG["agent"])
                self._log("智能体调用失败: %s" % e)
                return

            valid = r.get("valid", r.get("ok", False))
            text = r.get("text") or r.get("sentence") or ""
            tag = r.get("tag") or r.get("label") or ""
            err = r.get("error") or r.get("errorCode") or ""

            if valid:
                self.lbl_agent_text.config(text=text, fg="#9fe1a0")
                self.lbl_agent_tag.config(text=("标签: %s" % tag) if tag else "")
                self._log("智能体: %s  [%s]" % (text, tag))
                if CFG.get("push_polished_to_tft", True):
                    self._tft(r.get("short") or text)
                    try:
                        _post_json(CFG["backend"] + "/api/display", {
                            "source": "sign", "event": "final", "text": r.get("short") or text,
                            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                    except Exception:
                        pass
            else:
                self.lbl_agent_text.config(text="词汇杂乱，无法成句" + ("（%s）" % err if err else ""),
                                            fg="#e0b46b")
                self._log("智能体判定无效" + ("  原因: %s" % err if err else ""))
        threading.Thread(target=work, daemon=True).start()

    # ---------------- 自动拉起后台服务(双击软件就全起来) ----------------
    def _autostart_services(self):
        if not CFG.get("autostart_services", True):
            self._log("自动启动已关闭(config.json: autostart_services=false), 需手动开服务")
            return
        self.booting = True
        self.lbl_be.config(text="● 后端启动中…", fg="#e0c06b")
        self.lbl_agent.config(text="● 智能体启动中…", fg="#e0c06b")
        threading.Thread(target=self._boot_services, daemon=True).start()

    def _boot_services(self):
        """在后台线程里: 找项目根 -> 缺哪个起哪个 -> 等端口通 -> 更新顶栏"""
        def log(m):
            self.events.put(("log", m))

        def mark(which, text, color):
            self.events.put(("svc", which, text, color))

        root = _find_project_root()
        if not root:
            log("✗ 没找到项目目录(要含 web/app.py), 服务无法自动启动")
            mark("be", "● 后端离线", "#c46b6b")
            mark("agent", "● 智能体离线", "#8a8a8a")
            self.booting = False
            return
        log("项目目录: %s" % root)

        jobs = []      # [(显示名, 端口, 进程)]

        # ---- 1) Flask 后端 5000 ----
        if _port_open(5000):
            log("后端已在运行(5000), 跳过启动")
        else:
            py = _find_python()
            web = os.path.join(root, "web")
            if not py:
                log("✗ 没找到 python.exe, 无法自动启动后端(可在 config.json 填 python_exe)")
                mark("be", "● 后端离线", "#c46b6b")
            else:
                try:
                    p = _spawn([py, "app.py"], web, "backend.log")
                    self.procs["backend"] = p
                    jobs.append(("后端", 5000, p))
                    log("正在启动后端: %s (%s)" % (py, web))
                except Exception as e:
                    log("✗ 后端启动失败: %s" % e)
                    mark("be", "● 后端离线", "#c46b6b")

        # ---- 2) 智能体 8080 ----
        if _port_open(8080):
            log("智能体已在运行(8080), 跳过启动")
        else:
            java = _find_java()
            jar = os.path.join(root, "agent", "target", "wusheng-agent.jar")
            if not java:
                log("✗ 没找到 java.exe, 无法自动启动智能体(可在 config.json 填 java_exe)")
                mark("agent", "● 智能体离线", "#8a8a8a")
            elif not os.path.isfile(jar):
                log("✗ 缺少 agent\\target\\wusheng-agent.jar, 先到 agent 目录执行 mvn package")
                mark("agent", "● 智能体离线", "#8a8a8a")
            else:
                try:
                    p = _spawn([java, "-jar", jar], os.path.join(root, "agent"), "agent.log")
                    self.procs["agent"] = p
                    jobs.append(("智能体", 8080, p))
                    log("正在启动智能体: %s" % jar)
                except Exception as e:
                    log("✗ 智能体启动失败: %s" % e)
                    mark("agent", "● 智能体离线", "#8a8a8a")

        # ---- 3) 等端口就绪(后端几秒, 智能体约 8~15 秒) ----
        for name, port, proc in jobs:
            ready = False
            for _ in range(60):                     # 最多等 60*0.5 = 30 秒
                if _port_open(port):
                    ready = True
                    break
                if proc.poll() is not None:         # 进程自己退了, 别再干等
                    break
                time.sleep(0.5)
            if ready:
                log("✓ %s已就绪(:%d)" % (name, port))
            else:
                log("✗ %s启动超时, 看 logs\\%s.log" %
                    (name, "backend.log" if port == 5000 else "agent.log"))

        # 结果落实到顶栏(没起的那两个 job 也要给个明确状态)
        mark("be", "● 后端在线" if _port_open(5000) else "● 后端离线",
             "#5ddc9a" if _port_open(5000) else "#c46b6b")
        mark("agent", "● 智能体在线" if _port_open(8080) else "● 智能体离线",
             "#5ddc9a" if _port_open(8080) else "#8a8a8a")
        log("服务自检完成")
        if _port_open(5000):
            log("网站已就绪: 点下面「🌐 打开网站」就能看, 不用记地址")
        self.booting = False

    # ---------------- 服务状态轮询 ----------------
    def _check_services(self):
        if self.booting:
            self.root.after(3000, self._check_services)
            return

        def work():
            ok = backend_alive()
            self.lbl_be.config(text="● 后端在线" if ok else "● 后端离线",
                               fg="#5ddc9a" if ok else "#c46b6b")
            self.backend_ok = ok
            # 智能体探活: 用一次轻量请求
            try:
                req = urllib.request.Request(CFG["agent"].replace("/compose", "/health"),
                                             method="GET")
                urllib.request.urlopen(req, timeout=1.5).read()
                self.lbl_agent.config(text="● 智能体在线", fg="#5ddc9a")
            except Exception:
                self.lbl_agent.config(text="● 智能体离线", fg="#8a8a8a")
        threading.Thread(target=work, daemon=True).start()
        self.root.after(8000, self._check_services)

    def on_close(self):
        # 注意: 这里故意不杀后端/智能体进程 —— 关掉界面后网页和学习资源还能继续用,
        # 想彻底停掉请双击 项目目录\99_停止全部服务.bat
        if self.worker:
            self.worker.stop()
        if self.vworker:
            self.vworker.stop()
        self.root.destroy()


def main():
    print("[GUI] 正在加载识别模型…")
    core = load_core()
    print("[GUI] 模型已加载, 词库: %s" % " ".join(core._WORDS))
    root = tk.Tk()
    App(root, core)
    root.mainloop()


if __name__ == "__main__":
    main()
