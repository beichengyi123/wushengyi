package com.wusheng.agent.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * 智能体配置(对应 application.yml 里的 agent.*)
 * apiKey 只存在于本服务, ESP32 与桌面软件都拿不到。
 */
@ConfigurationProperties(prefix = "agent")
public class AgentProps {

    private final Deepseek deepseek = new Deepseek();
    private final Store store = new Store();

    public Deepseek getDeepseek() { return deepseek; }
    public Store getStore() { return store; }

    public static class Deepseek {
        /** DeepSeek 密钥(sk- 开头) */
        private String apiKey = "";
        /** OpenAI 兼容接口地址 */
        private String baseUrl = "https://api.deepseek.com/v1";
        private String model = "deepseek-chat";
        /** 单次调用超时(秒), 超时按异常返回 */
        private int timeoutSeconds = 20;
        private double temperature = 0.2;

        public String getApiKey() { return apiKey; }
        public void setApiKey(String apiKey) { this.apiKey = apiKey; }
        public String getBaseUrl() { return baseUrl; }
        public void setBaseUrl(String baseUrl) { this.baseUrl = baseUrl; }
        public String getModel() { return model; }
        public void setModel(String model) { this.model = model; }
        public int getTimeoutSeconds() { return timeoutSeconds; }
        public void setTimeoutSeconds(int timeoutSeconds) { this.timeoutSeconds = timeoutSeconds; }
        public double getTemperature() { return temperature; }
        public void setTemperature(double temperature) { this.temperature = temperature; }
    }

    public static class Store {
        /** 是否把处理结果写回网站数据库(通过 Flask 接口, 沿用原有存储逻辑) */
        private boolean enabled = true;
        /** Flask 网站地址 */
        private String backendUrl = "http://127.0.0.1:5000";
        private int timeoutSeconds = 5;

        public boolean isEnabled() { return enabled; }
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
        public String getBackendUrl() { return backendUrl; }
        public void setBackendUrl(String backendUrl) { this.backendUrl = backendUrl; }
        public int getTimeoutSeconds() { return timeoutSeconds; }
        public void setTimeoutSeconds(int timeoutSeconds) { this.timeoutSeconds = timeoutSeconds; }
    }
}
