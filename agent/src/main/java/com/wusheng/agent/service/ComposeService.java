package com.wusheng.agent.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.wusheng.agent.config.AgentProps;
import com.wusheng.agent.dto.ComposeResponse;
import dev.langchain4j.model.openai.OpenAiChatModel;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.List;

/**
 * 核心: 手语离散词 -> 通顺中文句子 + 分类标签 (LangChain4j 调用 DeepSeek)
 *
 * 智能体固定工作任务(对应需求 1~3、5):
 *   1) 接收词汇列表
 *   2) 调 DeepSeek 润色/理顺语序, 无法成句则判无效, 有效则打标签
 *   3) 固定 JSON 返回
 *   5) 超时/报错 -> 返回异常标识, 不抛 500 给调用方
 */
@Service
public class ComposeService {

    private static final Logger log = LoggerFactory.getLogger(ComposeService.class);

    /** 提示词: 约束只能用手语词、不许编造事实, 输出严格 JSON */
    private static final String SYSTEM_PROMPT = """
            你是"无声有译"手语翻译系统的语言助手。使用者是听障人士，用手语表达，
            对方(听人)通过屏幕上的文字来理解。
            用户给你的是一串由手语识别出的离散词，常见问题：语序颠倒、缺虚词、缺主语宾语、有重复。
            你的任务：
            1. 用这些词组成一句通顺、自然的现代汉语。手语语序(如"我 吃饭 去")要理顺为汉语语序；
               可补必要的虚词(的/了/在/吗等)和标点，可补全上下文里明显省略的主语或宾语，
               但绝对不能新增事实信息(不得加入词里没有的人名、地点、数字、时间、理由)。
            2. 如果给了"上文"，用它理解指代和省略(例如上文提到"医院"，本句"他 去"可整理为"他去医院")，
               但不要重复上文已说过的话，只输出本句。
            3. 判断交际意图，标签只能是三者之一：陈述、询问、问候。疑问句用问号，命令/请求也用陈述标签。
            4. 另外给一个"上屏短句"：不超过 12 个字，能放下一行小屏幕。
               短句必须逐字取自完整句子 text，只允许删掉句号/逗号等标点，
               绝对不能改字、换字或省略任何一个词；如果整句(去标点后)不超过 12 个字，
               short 就直接等于去掉句号逗号后的整句。
            5. 若这些词彼此无关、无法组成有意义的一句话，判定无效。
            只输出 JSON，不要解释，不要 markdown 代码块：
            有效：{"valid":true,"text":"整理后的完整句子。","short":"不超过12字的上屏短句","tag":"陈述"}
            无效：{"valid":false,"text":"","short":"","tag":""}
            示例1  输入: 我 / 要 / 去 / 学校        输出: {"valid":true,"text":"我要去学校。","short":"我要去学校","tag":"陈述"}
            示例2  输入: 你 / 名字 / 什么           输出: {"valid":true,"text":"你叫什么名字？","short":"你叫什么名字？","tag":"询问"}
            示例3  输入: 你 / 好 / 谢谢             输出: {"valid":true,"text":"你好，谢谢。","short":"你好，谢谢","tag":"问候"}
            示例4  输入: 吃饭 / 帮助 / 现在         输出: {"valid":true,"text":"我现在去吃饭，需要帮助吗？","short":"我现在去吃饭需要帮助吗？","tag":"询问"}
            """;

    private final AgentProps props;
    private final ObjectMapper mapper = new ObjectMapper();
    private OpenAiChatModel model;

    public ComposeService(AgentProps props) {
        this.props = props;
    }

    @PostConstruct
    public void init() {
        AgentProps.Deepseek d = props.getDeepseek();
        if (isBlank(d.getApiKey())) {
            log.warn("未配置 agent.deepseek.api-key, 接口会返回 KEY_MISSING");
            return;
        }
        // DeepSeek 与 OpenAI 协议兼容, 只换 baseUrl/modelName
        this.model = OpenAiChatModel.builder()
                .apiKey(d.getApiKey())
                .baseUrl(d.getBaseUrl())
                .modelName(d.getModel())
                .temperature(d.getTemperature())
                .timeout(Duration.ofSeconds(d.getTimeoutSeconds()))
                .maxRetries(1)
                .build();
        log.info("DeepSeek 模型就绪: {} @ {}", d.getModel(), d.getBaseUrl());
    }

    /** 主入口: 输入词列表, 输出固定结构 */
    public ComposeResponse compose(List<String> words) {
        return compose(words, null);
    }

    /** 带上下文的主入口: context 是最近几句成句, 用于理解指代/省略 */
    public ComposeResponse compose(List<String> words, List<String> context) {
        long t0 = System.currentTimeMillis();

        // --- 入参校验 ---
        if (words == null || words.isEmpty()) {
            return finish(ComposeResponse.invalid("EMPTY_INPUT"), t0);
        }
        if (model == null) {
            return finish(ComposeResponse.invalid("KEY_MISSING"), t0);
        }

        String joined = String.join(" / ", words);
        StringBuilder sb = new StringBuilder(SYSTEM_PROMPT);
        if (context != null && !context.isEmpty()) {
            sb.append("\n上文(最近几句, 仅用于理解指代, 不要重复输出): ")
              .append(String.join(" | ", context));
        }
        sb.append("\n输入: ").append(joined).append("\n输出:");
        String prompt = sb.toString();

        // --- 调模型, 超时/报错都转成异常标识 ---
        String answer;
        try {
            answer = model.generate(prompt);
        } catch (Exception e) {
            String msg = String.valueOf(e.getMessage()).toLowerCase();
            boolean timeout = msg.contains("timeout") || msg.contains("timed out")
                    || e.getClass().getSimpleName().toLowerCase().contains("timeout");
            log.warn("DeepSeek 调用失败({}): {}", timeout ? "TIMEOUT" : "UPSTREAM_ERROR", e.toString());
            return finish(ComposeResponse.invalid(timeout ? "TIMEOUT" : "UPSTREAM_ERROR"), t0);
        }

        // --- 解析模型返回的 JSON ---
        ComposeResponse r = parse(answer);
        log.info("词汇[{}] -> valid={} text={} tag={}", joined, r.isValid(), r.getText(), r.getTag());
        return finish(r, t0);
    }

    /** 从模型输出里抠出 JSON(模型偶尔会加 ```json 包裹或前后废话) */
    private ComposeResponse parse(String answer) {
        if (answer == null || answer.isBlank()) {
            return ComposeResponse.invalid("UPSTREAM_ERROR");
        }
        String s = answer.trim();
        int i = s.indexOf('{'), j = s.lastIndexOf('}');
        if (i < 0 || j <= i) {
            return ComposeResponse.invalid("UPSTREAM_ERROR");
        }
        try {
            JsonNode n = mapper.readTree(s.substring(i, j + 1));
            boolean valid = n.path("valid").asBoolean(false);
            String text = n.path("text").asText("").trim();
            String shortText = n.path("short").asText("").trim();
            String tag = n.path("tag").asText("").trim();
            if (!valid || text.isEmpty()) {
                return ComposeResponse.invalid(null);        // 模型判定"无法成句"
            }
            if (!tag.matches("陈述|询问|问候")) {
                tag = "陈述";                                 // 标签兜底, 不让脏标签进库
            }
            // 短句硬保证: 完整句去掉句号/逗号后若不超过 12 字, 直接用它作为上屏短句,
            // 保证屏幕显示与软件里的润色结果逐字一致, 不依赖模型自觉
            String stripped = stripPunct(text);
            if (stripped.length() <= 12) {
                shortText = stripped;
            } else if (shortText.isEmpty() || shortText.length() > 16) {
                // 整句太长且模型没给可用的短句(或太长)时才截断兜底
                shortText = stripped.length() > 12 ? stripped.substring(0, 12) : stripped;
            }
            ComposeResponse ok = ComposeResponse.ok(text, tag);
            ok.setShort(shortText);
            return ok;
        } catch (Exception e) {
            log.warn("模型返回无法解析: {}", answer);
            return ComposeResponse.invalid("UPSTREAM_ERROR");
        }
    }

    private ComposeResponse finish(ComposeResponse r, long t0) {
        r.setElapsedMs(System.currentTimeMillis() - t0);
        return r;
    }

    /** 去掉句中会占格但无实义的标点(句号/逗号/顿号/冒号等), 保留问号和感叹号 */
    private static String stripPunct(String s) {
        return s.replaceAll("[。，、；：,;:\\.\\s　]", "");
    }

    private boolean isBlank(String s) { return s == null || s.isBlank(); }
}
