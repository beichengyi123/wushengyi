# -*- coding: utf-8 -*-
# ================= 手语词 -> 自然语句 =================
# 作用: 把 1.py 识别出的一串手语词(如 你好 / 我 / 谢谢 / 你 / 再见),
#       组织成一句通顺的中文("你好，我谢谢你，再见。") 发到屏幕。
#
# 两种模式:
#   1) 有 API Key  -> 调大模型(DeepSeek, 兼容 OpenAI 格式), 语序和虚词更自然
#   2) 没 API Key  -> 用内置规则拼句, 不联网也能出句子(演示保底)
#
# 填 Key 的方法(约3分钟):
#   打开 https://platform.deepseek.com -> 登录 -> API keys -> 创建
#   复制 sk- 开头那串, 粘到下面 API_KEY 里
# ====================================================

import json
import re
import urllib.request

API_KEY = ""          # <- 填 DeepSeek 的 Key(sk- 开头), 留空则用规则拼句
BASE_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
TIMEOUT = 8

# 这些词本身就是完整一句, 拼句时前后要断开(注意"谢谢"不算: 常见"谢谢你"连用)
STANDALONE = {"你好", "再见", "对不起", "不客气", "没关系"}

SYSTEM_PROMPT = (
    "你是手语翻译助手。用户按手语动作顺序给你一组词，请把它们组织成一句自然、"
    "口语化的中文。要求：1) 只能用给定的词，允许调整顺序、补少量虚词(的/了/吗)和标点；"
    "2) 不要解释、不要加引号、不要换行；3) 总长不超过 16 个字；"
    "4) 只输出这句话本身。"
)


def rule_based(words):
    """不联网的保底拼句: 独立语前后断开, 其余直接相连"""
    if not words:
        return ""
    out = ""
    for i, w in enumerate(words):
        if w in STANDALONE:
            if out:
                out += "，"
            out += w
            if i < len(words) - 1:
                out += "，"
        else:
            out += w
    out = re.sub(r"，+", "，", out).strip("，")
    if not out:
        return ""
    return out + "。"


def _trim(text, limit=16):
    text = text.strip().strip('"“”').replace("\n", "")
    text = re.sub(r"\s+", "", text)
    return text[:limit]


def compose(words):
    """words: ['你好','我','谢谢','你','再见'] -> '你好，我谢谢你，再见。'"""
    words = [w for w in words if w]
    if not words:
        return ""
    if len(words) == 1:
        return words[0]
    if not API_KEY:
        return rule_based(words)
    try:
        body = json.dumps({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "、".join(words)},
            ],
            "temperature": 0.3,
            "max_tokens": 40,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            BASE_URL, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {API_KEY}"})
        resp = json.loads(urllib.request.urlopen(req, timeout=TIMEOUT).read())
        text = resp["choices"][0]["message"]["content"]
        text = _trim(text)
        return text if text else rule_based(words)
    except Exception as e:
        print(f"[sentence_api] 大模型调用失败, 改用规则拼句: {e}")
        return rule_based(words)


if __name__ == "__main__":
    for demo in (["你好", "我", "谢谢", "你", "再见"], ["我", "吃饭"], ["你", "好"]):
        print(demo, "->", compose(demo))
    print("Key 状态:", "已配置(走大模型)" if API_KEY else "未配置(规则拼句)")
