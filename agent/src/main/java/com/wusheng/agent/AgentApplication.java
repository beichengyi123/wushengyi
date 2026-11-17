package com.wusheng.agent;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import com.wusheng.agent.config.AgentProps;

/**
 * 无声有译 · 智能体服务入口
 * 端口默认 8080, 与 Flask 网站(5000)共存, 互不干扰。
 */
@SpringBootApplication
@EnableConfigurationProperties(AgentProps.class)
public class AgentApplication {
    public static void main(String[] args) {
        SpringApplication.run(AgentApplication.class, args);
    }
}
