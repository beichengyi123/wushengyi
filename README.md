# 无声有译 · WuShengYi

基于边缘 AI 视觉 + 物联网的手语双向智能翻译系统（大学生 AIoT 嵌入式系统创新大赛参赛作品）。

听障人士打手语 → 摄像头识别 → 屏幕显示润色后的句子；健听人对话筒说话 → 语音转文字 → ESP32 屏幕大字显示。**双向沟通，无需对方会手语。**

## 系统架构

```
 [摄像头]──> PC 桌面端 (MediaPipe 手势关键点 + kNN)
                 │                    │
                 │              [PC 麦克风/INMP441]
                 │                    │ 语音(PCM)
                 v                    v
          Flask Web 端(5000)    语音识别(百度API) / 局域网TCP直传
                 │                    │
                 v                    v
          智能体服务(8080) ──> DeepSeek 润色成通顺句子
                 │                    │
                 └────> MQTT(公网) <──┘
                          │
                          v
                 ESP32 + TFT 屏幕大字显示
```

## 核心功能

| 功能 | 说明 |
|---|---|
| 手语识别 | MediaPipe Hands 提取 21 个手部关键点，kNN 分类，支持 10+ 日常词汇 |
| 句子润色 | 离散词汇经 DeepSeek 大模型润色成自然句子（无 Key 时自动退回内置规则组句） |
| 语音转文字 | INMP441 I2S 麦克风采集 → 局域网 TCP 直传 PC（毫秒级）→ 百度短语音识别 → MQTT 回传屏幕 |
| 局域网直传 | PC 每 30s 将本机 IP 发布到 MQTT（retain），ESP32 动态订阅后 TCP 直连 5001 端口，语音延迟从 3~6s 降到 1s 级 |
| Web 中控 | Flask 局域网网页端，手机可访问，查看识别记录与采集管理 |

## 目录结构

```
├── 1.py                  # 主识别程序（手势识别 + 语音通道 + MQTT 上行）
├── sign_collect.py       # 手势样本采集
├── train_model.py        # kNN 模型训练
├── sign_recognize.py     # 识别核心
├── sign_guard.py         # 采集防抖/质量守护
├── voice_api.py          # 百度语音转文字封装（需填自己的 Key）
├── sentence_api.py       # DeepSeek 句子润色封装（需填自己的 Key）
├── sign_data/            # 手势样本数据集（按词汇分目录）
├── sign_model.npz        # 训练好的 kNN 模型
├── gui/                  # 桌面端（tkinter），PyInstaller 可打包为 exe
│   ├── gui_app.py        # 主界面 + 语音通道线程
│   └── build_exe.py      # 一键打包脚本
├── web/                  # Flask 局域网 Web 端
├── agent/                # 智能体服务（Spring Boot + LangChain4j + DeepSeek）
├── 固件与ESP32/           # ESP32 TFT 屏幕端 + INMP441 语音端固件（Arduino）
└── 启动说明_先看我.txt     # 快速启动说明
```

## 快速开始

### 环境要求
- Python 3.11（依赖：mediapipe, opencv-python, numpy, paho-mqtt, requests 等）
- JDK 17 + Maven（智能体服务，可选）
- Arduino IDE + ESP32 核心 3.3+（固件烧录，可选）

### 1. 手语识别主程序
```bash
python 1.py
```
首次运行会拉起摄像头窗口，对准手部即可识别。

### 2. 桌面端（推荐）
```bash
cd gui
python gui_app.py          # 直接运行
python build_exe.py        # 或打包成 exe
```

### 3. 智能体服务（可选，用于句子润色）
```bash
cd agent
# 先在 agent/src/main/resources/application.yml 填入你的 DeepSeek 密钥
mvn spring-boot:run        # 监听 8080
```

### 4. 语音转文字
在 `voice_api.py` 中填入你自己的百度语音 API Key / Secret Key（免费申请：https://ai.baidu.com → 语音技术 → 短语音识别）。

### 5. ESP32 固件
用 Arduino IDE 打开 `固件与ESP32/ESP32_INMP441_Voice_v5/`，填入 WiFi 名和密码后烧录。引脚配置见目录内说明文档。

## 自定义密钥（上传开源版已脱敏）

以下位置需要填入你自己的密钥才能使用对应云服务：
- `voice_api.py` → 百度语音 API_KEY / SECRET_KEY
- `agent/src/main/resources/application.yml` → DeepSeek api-key

不填也能跑：语音转文字不可用、句子润色退回内置规则组句，手语识别完全不受影响。

## MQTT 主题约定

| 主题 | 方向 | 内容 |
|---|---|---|
| `aiot/sign/gesture/demo` | PC → ESP32 | 识别出的手语词/句子 |
| `aiot/voice/text` | PC → ESP32 | 语音转文字结果 |
| `aiot/pc/info` | PC → ESP32 | PC 局域网 IP（retain，供 TCP 直连） |
| `aiot/voice/audio` | ESP32 → PC | INMP441 录音（PCM 8kHz，MQTT 备用通道） |

## 声明

本项目为大学生创新竞赛作品，用于学习交流。手语识别基于自采集的小规模数据集，词汇覆盖有限；正式无障碍场景请参考专业手语翻译方案。

## License

MIT
