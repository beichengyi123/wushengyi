package com.wusheng.agent.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.wusheng.agent.config.AgentProps;
import com.wusheng.agent.dto.ComposeResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 落库: 把"抓拍图 + 原始词汇 + 润色句子 + 标签"写进网站数据库。
 *
 * 做法说明(为什么这么设计):
 *   数据库是 Flask 网站用的 SQLite, 表结构 records 一直没变。
 *   为了"不改动原有存储逻辑", 这里不直接连 SQLite, 而是调 Flask 已有的
 *   写入入口新增的 /api/agent_record —— 记录的字段映射、时间戳格式、
 *   事务提交全部仍由 Flask 那边负责, 一个地方说了算。
 *   网站打不开时只记日志, 不影响智能体把句子返回给调用方。
 */
@Service
public class RecordStore {

    private static final Logger log = LoggerFactory.getLogger(RecordStore.class);
    private static final DateTimeFormatter FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private final AgentProps props;
    private final ObjectMapper mapper = new ObjectMapper();
    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(3)).build();

    public RecordStore(AgentProps props) {
        this.props = props;
    }

    public void save(List<String> words, ComposeResponse r, String image, String ts) {
        if (!props.getStore().isEnabled()) {
            return;
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("words", words);
        body.put("text", r.getText());
        body.put("tag", r.getTag());
        body.put("valid", r.isValid());
        body.put("image", image == null ? "" : image);
        body.put("ts", (ts == null || ts.isBlank()) ? LocalDateTime.now().format(FMT) : ts);

        try {
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(props.getStore().getBackendUrl() + "/api/agent_record"))
                    .timeout(Duration.ofSeconds(props.getStore().getTimeoutSeconds()))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(
                            mapper.writeValueAsString(body), java.nio.charset.StandardCharsets.UTF_8))
                    .build();
            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() / 100 != 2) {
                log.warn("落库失败 HTTP {}: {}", resp.statusCode(), resp.body());
            }
        } catch (Exception e) {
            // 网站没开也不影响智能体返回, 只提示
            log.warn("落库请求异常(网站是否已启动?): {}", e.toString());
        }
    }
}
