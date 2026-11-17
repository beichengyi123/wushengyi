package com.wusheng.agent.dto;

/**
 * 固定返回格式(需求第3条):
 *   valid : 是否有效
 *   text  : 润色后的文本
 *   tag   : 陈述 / 询问 / 问候 (无效时为空)
 *   error : 异常标识, 正常为 null
 *           TIMEOUT        调用 DeepSeek 超时
 *           UPSTREAM_ERROR 接口报错 / 返回无法解析
 *           EMPTY_INPUT    没收到词汇
 *           KEY_MISSING    后端没配置密钥
 */
public class ComposeResponse {

    private boolean valid;
    private String text = "";
    private String shortText = "";
    private String tag = "";
    private String error;
    private long elapsedMs;

    public ComposeResponse() {}

    public static ComposeResponse ok(String text, String tag) {
        ComposeResponse r = new ComposeResponse();
        r.valid = true;
        r.text = text;
        r.tag = tag;
        return r;
    }

    public static ComposeResponse invalid(String error) {
        ComposeResponse r = new ComposeResponse();
        r.valid = false;
        r.error = error;
        return r;
    }

    public boolean isValid() { return valid; }
    public void setValid(boolean valid) { this.valid = valid; }
    public String getText() { return text; }
    public void setText(String text) { this.text = text; }
    public String getTag() { return tag; }
    public void setTag(String tag) { this.tag = tag; }
    /** 上屏短句(<=12字, 给 ESP32 小屏幕用; 为空则前端退回用 text) */
    public String getShort() { return shortText; }
    public void setShort(String shortText) { this.shortText = shortText; }
    public String getError() { return error; }
    public void setError(String error) { this.error = error; }
    public long getElapsedMs() { return elapsedMs; }
    public void setElapsedMs(long elapsedMs) { this.elapsedMs = elapsedMs; }
}
