# -*- coding: utf-8 -*-
"""
无声有译 · 局域网 Web 中控 —— 后端 (Flask)

职责只有四件事(对应功能清单, 不额外扩展):
  1. 收识别结果(手语/语音) -> 存库 -> 同步给网页看板 + 终端(ESP32 轮询取)
  2. 置信度校验: 低于阈值的词不推送到硬件终端(网页仍显示, 打"已拦截"标记)
  3. 词库数据: 词 + 释义 + 配图 + 样本图
  4. 历史记录 + CSV 导出

数据来源说明(重要):
  - 1.py 不改。手语词级记录(带分数)由 bridge_mqtt.py 读 sign_log.txt 得到;
  - 屏幕文字(逐词拼接/整句/语音)由 bridge_mqtt.py 订阅 MQTT 得到;
  - 本文件只负责"接住并分发", 不做任何识别。

启动:  D:/python/python.exe web/app.py
访问:  http://<本机局域网IP>:5000
"""

import os
import io
import csv
import json
import time
import sqlite3
import threading
from datetime import datetime

from flask import Flask, request, jsonify, send_file, render_template, g

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)                 # 无声有译-项目 根目录
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "records.db")
CONF_PATH = os.path.join(DATA_DIR, "config.json")
GEST_DIR = os.path.join(BASE_DIR, "static", "gestures")
# 抓拍图存放目录(GUI 桌面软件上传, 网页历史页可直接引用)
SNAP_DIR = os.path.join(BASE_DIR, "static", "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)
WORDS_INFO = os.path.join(BASE_DIR, "words_info.json")
SIGN_DATA = os.path.join(PROJ_DIR, "sign_data")

HOST = "0.0.0.0"     # 0.0.0.0 = 局域网内其它设备能访问
PORT = 5000

DEFAULT_CONF = {
    "threshold": 0.50,   # 置信度阈值: 分数低于它的手语词不推送到 TFT
    "mode": "auto",      # auto = 识别结果自动上屏; single = 只有网页手动发才上屏
    "locked": False,     # True = 终端锁屏, 任何内容都不上屏
}

app = Flask(__name__)

# ---------------- 运行时状态(只存内存, 重启后屏幕内容清空) ----------------
_lock = threading.Lock()
_state = {
    "sign_line": "",      # 网页看板: 手语当前句(逐词拼接过程)
    "voice_text": "",     # 网页看板: 最近一条语音
    "terminal_sign": "",  # 终端: 手语区正在显示的内容
    "terminal_voice": "", # 终端: 语音区正在显示的内容
    "ver": 0,             # 版本号, ESP32 靠它判断要不要重画
}
_conf = dict(DEFAULT_CONF)


# ---------------- 配置读写 ----------------
def load_conf():
    global _conf
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CONF_PATH):
        try:
            with open(CONF_PATH, "r", encoding="utf-8") as fp:
                saved = json.load(fp)
            for k in DEFAULT_CONF:
                if k in saved:
                    _conf[k] = saved[k]
        except Exception as e:
            print("[配置] 读取失败, 用默认值:", e)
    print("[配置] 阈值=%.2f 模式=%s 锁屏=%s" % (_conf["threshold"], _conf["mode"], _conf["locked"]))


def save_conf():
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CONF_PATH, "w", encoding="utf-8") as fp:
        json.dump(_conf, fp, ensure_ascii=False, indent=2)


# ---------------- 数据库 ----------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, timeout=10)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            type     TEXT NOT NULL,      -- sign / voice
            text     TEXT,               -- 识别文本(手语: 词; 语音: 整句)
            word     TEXT,               -- 手语专用: 这个词
            score    REAL,               -- 手语专用: 置信度分数 0~1
            dist     REAL,               -- 手语专用: 到样本的距离
            th       REAL,               -- 手语专用: 当时的门槛
            hand     TEXT,               -- 手语专用: 单手/双手
            sentence TEXT,               -- 手语专用: 当时本句拼到哪
            blocked  INTEGER DEFAULT 0,  -- 1 = 低于阈值, 已拦截(没上 TFT)
            image    TEXT,               -- 抓拍图路径(预留: 1.py 存图后可填)
            ts       TEXT NOT NULL       -- 时间戳
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_type_ts ON records(type, ts)")
    # 升级新增: 智能体给句子打的标签(陈述/询问/问候)。
    # 只加一个可空列, 老数据不受影响, 原表结构与写入逻辑保持不变。
    try:
        con.execute("ALTER TABLE records ADD COLUMN tag TEXT")
    except Exception:
        pass
    con.commit()
    con.close()


def insert_record(**kw):
    db = get_db()
    db.execute("""
        INSERT INTO records(type, text, word, score, dist, th, hand, sentence, blocked, image, tag, ts)
        VALUES(:type, :text, :word, :score, :dist, :th, :hand, :sentence, :blocked, :image, :tag, :ts)
    """, {
        "type": kw.get("type", "sign"),
        "text": kw.get("text", ""),
        "word": kw.get("word"),
        "score": kw.get("score"),
        "dist": kw.get("dist"),
        "th": kw.get("th"),
        "hand": kw.get("hand"),
        "sentence": kw.get("sentence"),
        "blocked": int(kw.get("blocked", 0)),
        "image": kw.get("image"),
        "tag": kw.get("tag"),
        "ts": kw.get("ts") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    db.commit()


# ---------------- 工具 ----------------
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def bump():
    """终端内容变了就 +1, ESP32 只认这个号"""
    _state["ver"] += 1


def can_push():
    """终端能不能上屏: 没锁 && 自动模式"""
    return (not _conf["locked"]) and _conf["mode"] == "auto"


# ---------------- 桥接层写入接口 ----------------
@app.route("/api/sign_word", methods=["POST"])
def api_sign_word():
    """bridge_mqtt.py 从 sign_log.txt 读到一行 -> 存一条手语词记录
       低于阈值的仍然入库(可回溯), 但 blocked=1, 不推终端"""
    d = request.get_json(force=True, silent=True) or {}
    score = d.get("score")
    blocked = 1 if (score is not None and score < _conf["threshold"]) else 0
    insert_record(type="sign", text=d.get("word", ""), word=d.get("word"),
                  score=score, dist=d.get("dist"), th=d.get("th"),
                  hand=d.get("hand"), sentence=d.get("sentence"),
                  blocked=blocked, image=d.get("image"),
                  ts=d.get("ts") or now_str())
    return jsonify(ok=True, blocked=bool(blocked))


@app.route("/api/snapshot", methods=["POST"])
def api_snapshot():
    """GUI 桌面软件上传抓拍图 -> 存进 static/snapshots, 返回可直接访问的相对路径
       表单: file(图片) + word + ts
       说明: 只负责存文件, 记录入库仍走 /api/sign_word(沿用原表结构, 不改存储逻辑)"""
    f = request.files.get("file")
    if f is None:
        return jsonify(ok=False, msg="没有文件"), 400
    word = (request.form.get("word") or "shot").strip()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe_word = "".join(ch for ch in word if ch.isalnum() or ch in "_-") or "shot"
    name = "%s_%s.jpg" % (stamp, safe_word)
    f.save(os.path.join(SNAP_DIR, name))
    return jsonify(ok=True, path="snapshots/" + name, name=name)


@app.route("/api/agent_record", methods=["POST"])
def api_agent_record():
    """智能体服务(SpringBoot)处理完成后回调这里落库, 用的是原有表与原有写入函数。
       body: {words:[...], text:"润色后的句子", tag:"陈述", image:"snapshots/x.jpg", ts}
       字段映射(为了不改历史页展示逻辑, 复用 type='sign'):
         word     = 润色后的句子(历史页"词"列直接看到整句)
         sentence = 原始离散词汇(便于回溯智能体到底收到了什么)
         text     = 润色后的句子(CSV 里可读)
         tag      = 陈述/询问/问候
         image    = 抓拍图相对路径
       """
    d = request.get_json(force=True, silent=True) or {}
    words = d.get("words") or []
    if isinstance(words, str):
        words = [w for w in words.replace(",", " ").split() if w]
    text = (d.get("text") or "").strip()
    tag = (d.get("tag") or "").strip()
    if not text:
        return jsonify(ok=False, msg="没有润色文本"), 400
    insert_record(type="sign", text=text, word=text,
                  sentence=" ".join(words), tag=tag,
                  image=d.get("image") or "",
                  blocked=0, ts=d.get("ts") or now_str())
    return jsonify(ok=True)


@app.route("/api/display", methods=["POST"])
def api_display():
    """bridge_mqtt.py 收到 MQTT 屏幕文字 -> 更新看板 + 按规则决定是否推终端
       body: {source: sign|voice, text, event: word|final|clear|voice, score}"""
    d = request.get_json(force=True, silent=True) or {}
    source = d.get("source", "sign")
    event = d.get("event", "word")
    text = (d.get("text") or "").strip()
    score = d.get("score")

    with _lock:
        if source == "sign":
            _state["sign_line"] = "" if event == "clear" else text
            if can_push():
                if event == "clear":
                    _state["terminal_sign"] = ""
                    bump()
                elif event != "word":
                    # 整句/撤销重发: 词已经各自过过阈值, 整句直接上屏
                    _state["terminal_sign"] = text
                    bump()
                elif score is None or score >= _conf["threshold"]:
                    _state["terminal_sign"] = text
                    bump()
                # 否则: 该词低于阈值 -> 不推终端(网页看板上仍有这条, 带"已拦截"标记)
        else:
            if text:
                insert_record(type="voice", text=text, ts=d.get("ts") or now_str())
                _state["voice_text"] = text
                if can_push():
                    _state["terminal_voice"] = text
                    bump()
        ver = _state["ver"]
    return jsonify(ok=True, ver=ver)


# ---------------- 网页状态 ----------------
@app.route("/api/state")
def api_state():
    db = get_db()
    last_sign = db.execute("SELECT * FROM records WHERE type='sign' ORDER BY id DESC LIMIT 1").fetchone()
    last_voice = db.execute("SELECT * FROM records WHERE type='voice' ORDER BY id DESC LIMIT 1").fetchone()
    recent = db.execute("SELECT * FROM records ORDER BY id DESC LIMIT 12").fetchall()
    with _lock:
        st = dict(_state)
    return jsonify({
        "sign_line": st["sign_line"],
        "voice_text": st["voice_text"],
        "terminal": {"sign": st["terminal_sign"], "voice": st["terminal_voice"], "ver": st["ver"]},
        "conf": _conf,
        "last_sign": dict(last_sign) if last_sign else None,
        "last_voice": dict(last_voice) if last_voice else None,
        "recent": [dict(r) for r in recent],
    })


# ---------------- 终端控制 ----------------
@app.route("/api/terminal/send", methods=["POST"])
def api_terminal_send():
    """网页手动下发文字"""
    d = request.get_json(force=True, silent=True) or {}
    text = (d.get("text") or "").strip()
    target = d.get("target", "sign")          # sign / voice
    if not text:
        return jsonify(ok=False, msg="内容为空"), 400
    if _conf["locked"]:
        return jsonify(ok=False, msg="屏幕已锁定, 先解锁"), 409
    with _lock:
        if target == "voice":
            _state["terminal_voice"] = text
        else:
            _state["terminal_sign"] = text
        bump()
    return jsonify(ok=True)


@app.route("/api/terminal/lock", methods=["POST"])
def api_terminal_lock():
    d = request.get_json(force=True, silent=True) or {}
    _conf["locked"] = bool(d.get("locked"))
    save_conf()
    with _lock:
        bump()
    return jsonify(ok=True, locked=_conf["locked"])


@app.route("/api/terminal/mode", methods=["POST"])
def api_terminal_mode():
    d = request.get_json(force=True, silent=True) or {}
    mode = d.get("mode")
    if mode not in ("auto", "single"):
        return jsonify(ok=False, msg="mode 只能是 auto 或 single"), 400
    _conf["mode"] = mode
    save_conf()
    return jsonify(ok=True, mode=mode)


@app.route("/api/terminal/clear", methods=["POST"])
def api_terminal_clear():
    d = request.get_json(force=True, silent=True) or {}
    target = d.get("target", "both")
    with _lock:
        if target in ("sign", "both"):
            _state["terminal_sign"] = ""
        if target in ("voice", "both"):
            _state["terminal_voice"] = ""
        bump()
    return jsonify(ok=True)


@app.route("/api/config", methods=["POST"])
def api_config():
    d = request.get_json(force=True, silent=True) or {}
    if "threshold" in d:
        try:
            v = float(d["threshold"])
        except Exception:
            return jsonify(ok=False, msg="阈值必须是数字"), 400
        if not 0 <= v <= 1:
            return jsonify(ok=False, msg="阈值范围 0~1"), 400
        _conf["threshold"] = round(v, 2)
    save_conf()
    return jsonify(ok=True, conf=_conf)


# ---------------- ESP32 专用: 极简文本格式(不依赖 JSON 库) ----------------
@app.route("/api/terminal/esp")
def api_terminal_esp():
    """ESP32 每 1~2 秒轮询一次, 返回四行:
       VER:12       版本号(变了才重画)
       LOCK:0       1=锁屏
       S:我要去学校  手语区
       V:你好        语音区
       注: 内容里的换行/冒号已在写入前清理, 这里只做防护"""
    with _lock:
        st = dict(_state)
    def clean(s):
        return (s or "").replace("\n", " ").replace("\r", " ").replace(":", "：")
    body = "VER:%d\nLOCK:%d\nS:%s\nV:%s\n" % (
        st["ver"], 1 if _conf["locked"] else 0,
        clean(st["terminal_sign"]), clean(st["terminal_voice"]))
    return app.response_class(body, mimetype="text/plain; charset=utf-8")


# ---------------- 历史记录 / CSV ----------------
@app.route("/api/records")
def api_records():
    typ = request.args.get("type", "sign")
    limit = min(int(request.args.get("limit", 100)), 1000)
    if typ not in ("sign", "voice"):
        return jsonify(ok=False, msg="type 只能是 sign / voice"), 400
    rows = get_db().execute(
        "SELECT * FROM records WHERE type=? ORDER BY id DESC LIMIT ?", (typ, limit)).fetchall()
    return jsonify(ok=True, type=typ, rows=[dict(r) for r in rows])


@app.route("/api/export/<typ>.csv")
def api_export(typ):
    if typ not in ("sign", "voice"):
        return "type 只能是 sign / voice", 400
    rows = get_db().execute(
        "SELECT * FROM records WHERE type=? ORDER BY id ASC", (typ,)).fetchall()
    buf = io.StringIO()
    if typ == "sign":
        head = ["序号", "词", "置信度分数", "距最近样本", "门槛", "单/双手", "当时本句", "是否拦截", "时间", "抓拍图"]
    else:
        head = ["序号", "识别文本", "时间"]
    w = csv.writer(buf)
    w.writerow(head)
    for i, r in enumerate(rows, 1):
        if typ == "sign":
            w.writerow([i, r["word"], "%.2f" % r["score"] if r["score"] is not None else "",
                        "%.1f" % r["dist"] if r["dist"] is not None else "",
                        "%.1f" % r["th"] if r["th"] is not None else "",
                        r["hand"] or "", r["sentence"] or "",
                        "是" if r["blocked"] else "否", r["ts"],
                        r["image"] or ""])
        else:
            w.writerow([i, r["text"], r["ts"]])
    data = buf.getvalue()
    name = "手语识别记录_%s.csv" % datetime.now().strftime("%Y%m%d_%H%M") if typ == "sign" \
        else "语音转写记录_%s.csv" % datetime.now().strftime("%Y%m%d_%H%M")
    # utf-8-sig: Excel 直接打开不乱码
    return send_file(io.BytesIO(data.encode("utf-8-sig")), mimetype="text/csv",
                     as_attachment=True, download_name=name)


# ---------------- 词库 ----------------
@app.route("/api/words")
def api_words():
    info = {}
    if os.path.exists(WORDS_INFO):
        with open(WORDS_INFO, "r", encoding="utf-8") as fp:
            info = json.load(fp)
    words = []
    if os.path.isdir(SIGN_DATA):
        for w in sorted(os.listdir(SIGN_DATA)):
            d = os.path.join(SIGN_DATA, w)
            if not os.path.isdir(d):
                continue
            npy = [f for f in os.listdir(d) if f.lower().endswith(".npy")]
            img = "gestures/%s.png" % w
            samples = []
            for k in (1, 2, 3):
                p = "gestures/%s_s%d.png" % (w, k)
                if os.path.exists(os.path.join(GEST_DIR, "%s_s%d.png" % (w, k))):
                    samples.append(p)
            words.append({
                "word": w,
                "meaning": (info.get(w) or {}).get("meaning", ""),
                "count": len(npy),
                "image": img if os.path.exists(os.path.join(GEST_DIR, "%s.png" % w)) else "",
                "samples": samples,
            })
    return jsonify(ok=True, words=words)


# ---------------- 页面 ----------------
@app.route("/")
def page_index():
    return render_template("index.html", nav="dash")


@app.route("/words")
def page_words():
    return render_template("words.html", nav="words")


@app.route("/history")
def page_history():
    return render_template("history.html", nav="history")


@app.route("/guide")
def page_guide():
    return render_template("guide.html", nav="guide")


# ---------------- 手语样本采集（新增, 原有路由全部保持不变） ----------------
# collect_api.py 里是采集页 / 标签 / 上传 / 自检 / 入库的整套接口,
# 挂在同一个 5000 端口上, 不新起服务。
from collect_api import bp as collect_bp          # noqa: E402
app.register_blueprint(collect_bp)


if __name__ == "__main__":
    load_conf()
    init_db()
    print("=" * 52)
    print(" 无声有译 中控后端: http://%s:%d" % (HOST, PORT))
    print(" 局域网内用 http://<本机IP>:%d 访问" % PORT)
    print("=" * 52)
    # threaded=True: 网页轮询和 ESP32 轮询同时来, 不能互相堵
    app.run(host=HOST, port=PORT, threaded=True, debug=False)
