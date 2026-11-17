package com.wusheng.agent.dto;

/**
 * 请求体: 电脑桌面软件 / ESP32 提交过来的手语离散词汇
 * 例: {"words":["我","要","去","学校"], "image":"snapshots/20261107_2210_学校.jpg"}
 */
public class ComposeRequest {

    /** 词汇列表(必填) */
    private java.util.List<String> words;
    /** 原始字符串形式(可选, 便于日志排查) */
    private String raw;
    /** 抓拍图相对路径(可选, 由桌面软件抓拍上传后带回) */
    private String image;
    /** 上文最近几句(可选): 让智能体知道对话背景, 能正确理解"他/它/这个"等指代 */
    private java.util.List<String> context;
    /** 时间戳(可选, 不传则服务端补) */
    private String ts;

    public java.util.List<String> getWords() { return words; }
    public void setWords(java.util.List<String> words) { this.words = words; }
    public String getRaw() { return raw; }
    public void setRaw(String raw) { this.raw = raw; }
    public String getImage() { return image; }
    public void setImage(String image) { this.image = image; }
    public java.util.List<String> getContext() { return context; }
    public void setContext(java.util.List<String> context) { this.context = context; }
    public String getTs() { return ts; }
    public void setTs(String ts) { this.ts = ts; }
}
