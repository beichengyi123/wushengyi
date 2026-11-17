package com.wusheng.agent.controller;

import com.wusheng.agent.dto.ComposeRequest;
import com.wusheng.agent.dto.ComposeResponse;
import com.wusheng.agent.service.ComposeService;
import com.wusheng.agent.service.RecordStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * 智能体对外接口(只暴露两个, 保持简单):
 *   POST /api/agent/compose  手语词 -> 润色句子 + 标签 (+自动落库)
 *   GET  /api/agent/health   探活(桌面软件用它显示"智能体在线/离线")
 */
@RestController
@RequestMapping("/api/agent")
public class AgentController {

    private static final Logger log = LoggerFactory.getLogger(AgentController.class);

    private final ComposeService composeService;
    private final RecordStore recordStore;

    public AgentController(ComposeService composeService, RecordStore recordStore) {
        this.composeService = composeService;
        this.recordStore = recordStore;
    }

    @PostMapping("/compose")
    public ComposeResponse compose(@RequestBody ComposeRequest req) {
        List<String> words = req.getWords();
        // 兼容只传 raw 字符串的情况: "我 要 去 学校"
        if ((words == null || words.isEmpty()) && req.getRaw() != null && !req.getRaw().isBlank()) {
            words = List.of(req.getRaw().trim().split("[\\s,/、]+"));
        }
        ComposeResponse r = composeService.compose(words, req.getContext());

        // 有效才落库(抓拍图 + 原始词汇 + 润色句子 + 标签)
        if (r.isValid()) {
            recordStore.save(words, r, req.getImage(), req.getTs());
        } else {
            log.info("无效结果不落库: {}", r.getError());
        }
        return r;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        return Map.of("status", "UP", "service", "wusheng-agent", "model", "deepseek-chat");
    }
}
