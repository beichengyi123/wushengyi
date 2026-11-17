// ================= 无声有译 · ESP32 终端（语音 v5 · INMP441 I2S 数字麦克风） =================
// 相比 v4(ESP32_Voice_Final.ino) 的唯一变化: 模拟麦克风(GPIO34 ADC) -> INMP441 I2S 数字麦克风
// 屏幕逻辑 / MQTT 通道 / 自动断句 VAD / 按键模式 全部与 v4 完全一致。
//
// INMP441 接线(共6根):
//   VDD    -> 3.3V        (严禁接5V!)
//   GND    -> GND
//   L/R    -> GND         (接地=左声道)
//   SCK    -> GPIO14      (BCLK 位时钟)
//   WS     -> GPIO13      (LRCLK 左右声道选择)
//   SD     -> GPIO26      (DOUT 数据输出 -> ESP32 输入)
//
// 屏幕配置由 User_Setup.h 提供（ILI9341, CS=15 DC=27 MOSI=23 SCK=18 BL=32）
// 电脑端 1.py / voice_api.py 无需任何改动: 板内把 I2S 16kHz 数据降成 8kHz PCM,
// 和原来模拟麦的数据格式完全一样, 百度识别/上屏链路原样复用。
//
// USE_BUTTON = 0 : 自动断句(默认), 听到人声才录, 停顿0.4秒自动上传
// USE_BUTTON = 1 : 按键模式, 按住说话, 松开发送
//   按键一脚 -> GPIO4, 对角脚 -> GND (GPIO4 内部上拉, 按下=LOW)

#include <WiFi.h>
#include <PubSubClient.h>
#include <ESP_I2S.h>            // Arduino-ESP32 3.x 自带 I2S 库(老 driver/i2s.h 在 3.x 已移除)
#include <TFT_eSPI.h>
#include <U8g2_for_TFT_eSPI.h>

// ================= 配置区 =================
const char* WIFI_SSID     = "1";
const char* WIFI_PASSWORD = "你的WiFi密码";
const char* MQTT_BROKER   = "broker.hivemq.com";   // 2026-10-26 emqx超时, 换HiveMQ

const char* TOPIC_SIGN_TEXT  = "aiot/sign/gesture/demo";
const char* TOPIC_AUDIO_SEND = "aiot/voice/audio";
const char* TOPIC_AUDIO_TEXT = "aiot/voice/text";
const char* TOPIC_PC_INFO    = "aiot/pc/info";      // 电脑发布的局域网 IP(保留消息)

// ---- 局域网直传(比走公网 MQTT 快 3 秒左右) ----
// 电脑软件启动后会把 IP 发到 TOPIC_PC_INFO, 收到就改走 TCP 直连; 收不到则自动回退 MQTT
#define PC_TCP_PORT   5001
#define PC_CONNECT_MS 600          // 连不上电脑时最多等 0.6 秒, 然后回退

#define USE_BUTTON 0              // 0=自动断句模式(当前使用); 1=按键说话模式
#define PIN_BUTTON 4              // 外接按键: 一脚->GPIO4, 对角脚->GND
#define PIN_BUTTON2 0             // 板载 BOOT 键
#define BTN_DEBUG 0               // 1=每秒打印候选引脚电平(接线排查用); 调好后改 0

// ---- INMP441 I2S 接线 ----
#define I2S_WS_PIN   13            // WS(LRCLK)
#define I2S_BCK_PIN  14            // SCK(BCLK)
#define I2S_DATA_PIN 26            // SD(DOUT), L/R 接 GND = 左声道

#define I2S_READ_RATE   16000      // INMP441 以 16kHz 读(数字麦在 8kHz 时钟不稳, 读 16k 再降)
#define INMP_SHIFT      16         // 32位采样右移位数: 数字越小音量越大(16=标准, 说话偏小改14/12, 爆音改18)
// ==========================================

WiFiClient espClient;
WiFiClient tcpClient;         // 直连电脑的音频通道
String pcHost = "";           // 电脑局域网 IP, 由 TOPIC_PC_INFO 下发
PubSubClient mqtt(espClient);
TFT_eSPI tft = TFT_eSPI();
U8g2_for_TFT_eSPI u8f;
I2SClass i2s;                 // 麦克风 I2S 总线

// ---- 录音参数(输出仍是 8kHz/16bit, 与 v4 完全一致) ----
#define SAMPLE_RATE     8000
#define MAX_SECONDS     3                              // 单句最长 3 秒(超过强制发)
                                                       // 注意: 别改回 4 秒! 4 秒=64KB 缓存, ESP32 静态内存不够,
                                                       // 会报 "section .dram0.bss will not fit in region dram0_0_seg"
#define TOTAL_SAMPLES   (SAMPLE_RATE * MAX_SECONDS)     // 24000 点 = 48KB 静态
#define CHUNK_SAMPLES   800                             // 每 0.1 秒一个判断块(8kHz)
#define READ_PER_CHUNK  (CHUNK_SAMPLES * 2)             // 16kHz 读 1600 点 -> 降采样成 800 点
#define PRE_SAMPLES     4000                            // 0.5 秒预录, 防止吃掉第一个字
#define QUIET_CHUNKS    4                               // 连续 0.4 秒无声 = 说完了
#define MIC_DEBUG       1                               // 1=每秒打印麦克风电平(排查用); 正常后改0

// ---- 开口判定参数(能量RMS + 自适应底噪, 与 v4 相同思路) ----
// INMP441 是数字麦, 底噪远低于模拟麦, 下限值相应调低
#define VOICE_RMS_K     2.2f    // 开口: RMS 超过底噪的多少倍
#define VOICE_RMS_MIN   80.0f   // 开口绝对下限(数字麦静音10~20, 环境噪音40~50, 真人说话200+)
#define LOUD_CHUNKS     2       // 连续几块(每块0.1秒)超阈值才算开口 -> 短促噪音不触发
#define QUIET_RMS_K     1.35f   // 说完: RMS 回落到这个倍数以内
#define QUIET_RMS_MIN   30.0f
#define TARGET_RMS      600.0f  // 整句统一放大到这个响度再上传
#define NOISE_REJECT_RMS 150.0f // 整句平均RMS低于此值=只是噪音, 不上传(真人说话200~600)
#define MAX_GAIN        8.0f    // 放大上限, 防止把纯噪声放大爆音

static int16_t pcmBuffer[TOTAL_SAMPLES];
static int16_t preBuf[PRE_SAMPLES];
static uint32_t sampleIdx = 0;
static uint32_t voiceCount = 0;

static int   micPeakShow = 0;
static float micThShow = 0;

static bool speaking = false;
static int  quietRun = 0;
static int  loudRun = 0;
static float noiseRms = 40.0f;        // 底噪能量(RMS), 主判据
static float chunkRms = 0.0f;         // 最近一块的 RMS(去直流后)

unsigned long lastWifiRetry = 0;
unsigned long lastMqttRetry = 0;
const unsigned long RETRY_INTERVAL = 5000;

// ================= 屏幕显示函数 =================
void showSignText(const String &text) {
  tft.fillRect(0, 30, 320, 90, TFT_BLACK);
  u8f.setFont(u8g2_font_wqy16_t_gb2312);
  u8f.setForegroundColor(TFT_YELLOW);
  u8f.drawUTF8(20, 90, text.c_str());
}

void showVoiceText(const String &text) {
  tft.fillRect(0, 151, 320, 89, TFT_BLACK);
  u8f.setFont(u8g2_font_wqy16_t_gb2312);
  u8f.setForegroundColor(TFT_CYAN);
  u8f.drawUTF8(20, 200, text.c_str());
}

// ================= MQTT 回调 =================
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String text = "";
  for (unsigned int i = 0; i < length; i++) text += (char)payload[i];
  text.trim();
  if (text.length() == 0) return;

  if (String(topic) == TOPIC_SIGN_TEXT) {
    showSignText(text);
  } else if (String(topic) == TOPIC_AUDIO_TEXT) {
    showVoiceText(text);
  } else if (String(topic) == TOPIC_PC_INFO) {
    // 电脑把局域网 IP 发过来了 -> 以后的音频直接连它, 不再绕公网
    if (text != pcHost) {
      pcHost = text;
      Serial.printf("[局域网] 收到电脑地址 %s:%d, 音频改走直连\n", pcHost.c_str(), PC_TCP_PORT);
    }
  }
}

// ================= I2S 初始化（Arduino-ESP32 3.x: ESP_I2S 类） =================
void micInit() {
  i2s.setPins(I2S_BCK_PIN, I2S_WS_PIN, -1, I2S_DATA_PIN);   // bclk, ws, dout(不用), din
  // 主模式 / 16kHz / 32位槽 / 单声道 / 只取左槽（INMP441 的 L/R 接 GND = 左声道）
  if (!i2s.begin(I2S_MODE_STD, I2S_READ_RATE, I2S_DATA_BIT_WIDTH_32BIT,
                 I2S_SLOT_MODE_MONO, I2S_STD_SLOT_LEFT)) {
    Serial.println("!!! I2S 初始化失败: 检查 INMP441 的 VDD=3.3V / SCK=14 / WS=13 / SD=26 / L-R=GND");
  }

  // 上电后 DMA 里有几段垃圾数据, 全部丢掉
  static int32_t trash[256];
  for (int i = 0; i < 4; i++) {
    i2s.readBytes((char*)trash, sizeof(trash));
  }
}

// ================= 采一个 0.1 秒块 =================
// I2S 读 1600 个 32bit 采样(16kHz), 两两平均降成 800 个 int16(8kHz) 写入 dst
// readBytes 会阻塞到数据凑够, 天然按 0.1 秒节奏返回, 不需要手工对齐采样时钟
// 每块单独去直流(减块内均值), 更新 chunkRms(人声强度)
void sampleChunk(int16_t* dst) {
  static int32_t raw[READ_PER_CHUNK];
  size_t got = i2s.readBytes((char*)raw, sizeof(raw));
  int n = got / 4;
  if (n < 2) { chunkRms = 0; if (dst) memset(dst, 0, CHUNK_SAMPLES * 2); return; }

  int32_t sum = 0;
  for (int i = 0; i < n; i++) sum += (raw[i] >> INMP_SHIFT);

  int mean = (int)(sum / n);
  int peak = 0;
  float sumSq = 0.0f;
  int out = 0;
  for (int i = 0; i + 1 < n; i += 2) {
    // 两两平均 = 简易低通, 防降采样混叠
    int32_t v = ((raw[i] + raw[i + 1]) >> 1) - mean;
    int32_t s = v >> INMP_SHIFT;
    if (s > 32000) s = 32000;
    if (s < -32000) s = -32000;
    int a = s < 0 ? -s : s;
    if (a > peak) peak = a;
    sumSq += (float)s * (float)s;
    if (dst) dst[out] = (int16_t)s;
    out++;
    if (out >= CHUNK_SAMPLES) break;
  }
  chunkRms = sqrtf(sumSq / (float)(out ? out : 1));
  if (out < CHUNK_SAMPLES && dst) {
    memset(dst + out, 0, (CHUNK_SAMPLES - out) * 2);
  }
}

// 局域网直传: 4字节长度(大端) + PCM16 数据。成功返回 true
bool sendDirect(const uint8_t* buf, uint32_t len) {
  if (pcHost.length() == 0) return false;
  if (!tcpClient.connected()) {
    tcpClient.stop();
    if (!tcpClient.connect(pcHost.c_str(), PC_TCP_PORT, PC_CONNECT_MS)) return false;
  }
  uint8_t hdr[4];
  hdr[0] = (uint8_t)(len >> 24);
  hdr[1] = (uint8_t)(len >> 16);
  hdr[2] = (uint8_t)(len >> 8);
  hdr[3] = (uint8_t)(len);
  size_t w1 = tcpClient.write(hdr, 4);
  size_t w2 = tcpClient.write(buf, len);
  tcpClient.flush();
  if (w1 != 4 || w2 != len) {      // 半路断了: 关掉, 下一句自动重连
    tcpClient.stop();
    return false;
  }
  return true;
}

// 上传整句(先做响度归一, 保证电脑端音量门槛稳定通过)
void publishUtterance() {
  if (sampleIdx < (uint32_t)(SAMPLE_RATE / 4)) {   // 短于 0.25 秒当误触发丢掉
    Serial.printf("[语音] 太短(%.2f秒), 丢弃\n", (float)sampleIdx / SAMPLE_RATE);
    sampleIdx = 0;
    speaking = false;
    quietRun = 0;
    return;
  }
  // ---- 响度归一: 算整句RMS(去直流), 放大到 TARGET_RMS 附近 ----
  int32_t sum = 0;
  for (uint32_t i = 0; i < sampleIdx; i++) sum += pcmBuffer[i];
  int mean = (int)(sum / sampleIdx);
  float sumSq = 0;
  for (uint32_t i = 0; i < sampleIdx; i++) {
    float d = (float)(pcmBuffer[i] - mean);
    sumSq += d * d;
  }
  float rms = sqrtf(sumSq / sampleIdx);

  // ---- 底噪误触发拦截: 没有真正的说话音量就直接丢掉, 不浪费网络也不占用识别 ----
  // (数字麦静音 RMS 只有 10~20, 环境噪音偶尔冲到 40~50; 真人说话 200 以上)
  if (rms < NOISE_REJECT_RMS) {
    Serial.printf("[语音] 底噪误触发(RMS %.0f), 丢弃\n", rms);
    sampleIdx = 0;
    speaking = false;
    quietRun = 0;
    return;
  }

  float gain = (rms > 1.0f) ? TARGET_RMS / rms : 1.0f;
  if (gain > MAX_GAIN) gain = MAX_GAIN;
  if (gain < 1.0f) gain = 1.0f;                    // 已经够响就不压低, 保留原始动态
  if (gain > 1.05f) {
    for (uint32_t i = 0; i < sampleIdx; i++) {
      float v = (float)(pcmBuffer[i] - mean) * gain;   // 顺带把块间直流差也抹平
      if (v > 32000) v = 32000;
      if (v < -32000) v = -32000;
      pcmBuffer[i] = (int16_t)v;
    }
  }

  voiceCount++;
  uint32_t bytes = sampleIdx * sizeof(int16_t);
  Serial.printf("[语音 #%u] 整句 %.2f 秒 %u 字节, 归一增益 x%.1f\n",
                voiceCount, (float)sampleIdx / SAMPLE_RATE, bytes, gain);
  if (sendDirect((const uint8_t*)pcmBuffer, bytes)) {
    Serial.printf("[局域网] 直传电脑 成功 (%u 字节)\n", bytes);
  } else if (mqtt.connected()) {
    bool ok = mqtt.publish(TOPIC_AUDIO_SEND, (const uint8_t*)pcmBuffer, bytes);
    Serial.printf("[MQTT] 回退公网 %s (慢, 检查电脑是否开着)\n", ok ? "成功" : "失败!");
  } else {
    Serial.println("[MQTT] 未连接, 本句丢弃");
  }
  sampleIdx = 0;
  speaking = false;
  quietRun = 0;
}

// ================= 网络重连 =================
void ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  unsigned long now = millis();
  if (now - lastWifiRetry < RETRY_INTERVAL) return;
  lastWifiRetry = now;
  WiFi.disconnect();
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

void ensureMQTT() {
  if (WiFi.status() != WL_CONNECTED) return;
  if (mqtt.connected()) return;
  unsigned long now = millis();
  if (now - lastMqttRetry < RETRY_INTERVAL) return;
  lastMqttRetry = now;
  Serial.println("尝试连接 MQTT...");
  if (mqtt.connect("ESP32_Final_BiChannel")) {
    Serial.println("MQTT 成功");
    mqtt.subscribe(TOPIC_SIGN_TEXT);
    mqtt.subscribe(TOPIC_AUDIO_TEXT);
    mqtt.subscribe(TOPIC_PC_INFO);       // 电脑的局域网地址(保留消息, 订阅即到)
  }
}

// ================= Setup =================
void setup() {
  Serial.begin(115200);
  delay(200);

  // --- 麦克风 I2S 初始化 ---
  micInit();
#if USE_BUTTON
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  pinMode(PIN_BUTTON2, INPUT_PULLUP);
  Serial.printf("[麦克风] INMP441 I2S 就绪 (BCLK=%d WS=%d SD=%d), 按键说话模式\n", I2S_BCK_PIN, I2S_WS_PIN, I2S_DATA_PIN);
#else
  Serial.printf("[麦克风] INMP441 I2S 就绪 (BCLK=%d WS=%d SD=%d), 自动断句模式\n", I2S_BCK_PIN, I2S_WS_PIN, I2S_DATA_PIN);
#endif

  // --- 屏幕 ---
  tft.init();
  tft.setRotation(1);
  tft.fillScreen(TFT_BLACK);
  u8f.begin(tft);
  u8f.setFontMode(1);

  tft.fillRect(0, 0, 320, 30, TFT_NAVY);
  tft.setTextColor(TFT_WHITE);
  tft.setTextSize(2);
  tft.setCursor(10, 6);
  tft.print("Sign Translation");

  tft.drawFastHLine(0, 120, 320, TFT_LIGHTGREY);
  tft.fillRect(0, 121, 320, 30, TFT_DARKGREEN);
  tft.setCursor(10, 126);
  tft.print("Voice Transcription");

  u8f.setFont(u8g2_font_wqy16_t_gb2312);
  u8f.setForegroundColor(TFT_DARKGREY);
  u8f.drawUTF8(20, 90, "等待手语...");
  u8f.drawUTF8(20, 200, "等待语音...");

  // --- 网络 ---
  WiFi.mode(WIFI_STA);
  ensureWiFi();

  mqtt.setBufferSize(65000);   // 注意: 库参数是16位整数(最大65535), 写64*1024会溢出归零!
  mqtt.setServer(MQTT_BROKER, 1883);
  mqtt.setCallback(mqttCallback);
}

// ================= Loop =================
void loop() {
  ensureWiFi();
  ensureMQTT();
  mqtt.loop();

#if USE_BUTTON
  // ---- 按键模式: 按住说话, 松开发送(防抖逻辑与 v4 相同) ----
  static bool pressed = false;
  static uint8_t db = 0;
  bool raw = (digitalRead(PIN_BUTTON) == LOW) || (digitalRead(PIN_BUTTON2) == LOW);

  if (raw != pressed) {
    if (++db >= 3) { pressed = raw; db = 0; }
  } else {
    db = 0;
  }

  if (pressed && !speaking) {
    speaking = true;
    sampleIdx = 0;
    quietRun = 0;
    Serial.println("[语音] 按下, 录音中... (松开发送, 单段最长3秒)");
  } else if (!pressed && speaking) {
    Serial.println("[语音] 松开, 发送");
    publishUtterance();
  }

  if (speaking) {
    if (sampleIdx + CHUNK_SAMPLES <= TOTAL_SAMPLES) {
      sampleChunk(pcmBuffer + sampleIdx);
      sampleIdx += CHUNK_SAMPLES;
    } else {
      Serial.println("[语音] 超过3秒, 分段发送");
      publishUtterance();
    }
  }
#else
  if (!speaking) {
    // 待机: 预录缓冲整体前移一块, 新块写到末尾(保证时间顺序)
    memmove(preBuf, preBuf + CHUNK_SAMPLES,
            (PRE_SAMPLES - CHUNK_SAMPLES) * sizeof(int16_t));
    sampleChunk(preBuf + PRE_SAMPLES - CHUNK_SAMPLES);
    float rms = chunkRms;

    // 底噪跟随: 只在"还算安静"时更新, 速度放慢
    if (rms < noiseRms * 2.0f) {
      noiseRms = noiseRms * 0.95f + rms * 0.05f;
      if (noiseRms < 15.0f) noiseRms = 15.0f;
    }

    float th = noiseRms * VOICE_RMS_K;
    if (th < VOICE_RMS_MIN) th = VOICE_RMS_MIN;

    micPeakShow = (int)rms;
    micThShow = th;

#if MIC_DEBUG
    static unsigned long lastMicLog = 0;
    if (millis() - lastMicLog > 1000) {
      lastMicLog = millis();
      Serial.printf("[麦克风] RMS %.0f | 底噪 %.0f | 阈值 %.0f\n", rms, noiseRms, th);
    }
#endif

    // 连续 LOUD_CHUNKS 块超阈值才算开口
    if (rms > th) {
      loudRun++;
    } else {
      loudRun = 0;
    }

    if (loudRun >= LOUD_CHUNKS) {
      speaking = true;
      quietRun = 0;
      loudRun = 0;
      memcpy(pcmBuffer, preBuf, PRE_SAMPLES * sizeof(int16_t));
      sampleIdx = PRE_SAMPLES;
      Serial.printf("[语音] 检测到开口 (RMS %.0f, 底噪 %.0f)\n", rms, noiseRms);
    }
  } else {
    sampleChunk(pcmBuffer + sampleIdx);
    sampleIdx += CHUNK_SAMPLES;
    float rms = chunkRms;

    float quietTh = noiseRms * QUIET_RMS_K;
    if (quietTh < QUIET_RMS_MIN) quietTh = QUIET_RMS_MIN;

    if (rms < quietTh) {
      quietRun++;
      noiseRms = noiseRms * 0.9f + rms * 0.1f;
      if (noiseRms < 15.0f) noiseRms = 15.0f;
    } else {
      quietRun = 0;
    }

    if (quietRun >= QUIET_CHUNKS || sampleIdx >= TOTAL_SAMPLES) {
      publishUtterance();
    }
  }
#endif

  // 屏幕下方实时显示麦克风电平: 说话时数字必须明显变大
  static unsigned long lastMicDraw = 0;
  if (millis() - lastMicDraw > 1000) {
    lastMicDraw = millis();
    char mbuf[40];
    snprintf(mbuf, sizeof(mbuf), "声音 %.0f / 门槛 %.0f", (float)micPeakShow, micThShow);
    tft.fillRect(0, 214, 320, 26, TFT_BLACK);
    u8f.setFont(u8g2_font_wqy16_t_gb2312);
    u8f.setForegroundColor(micPeakShow > micThShow ? TFT_GREEN : TFT_DARKGREY);
    u8f.drawUTF8(20, 234, mbuf);
  }

  delay(2);
}
