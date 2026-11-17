// ================= 无声有译 · ESP32 终端（语音 v4 · 按键说话/自动断句 双模式） =================
// 屏幕逻辑与 MQTT 通道与原版完全一致。
// USE_BUTTON = 1 : 按键模式, 按住说话, 松开发送(最准, 没有底噪误触发)
// USE_BUTTON = 0 : 自动断句(v3), 听到人声才录, 停顿0.4秒自动上传
//
// 按键接法(两根杜邦线, 无需外接电阻):
//   按键一脚 -> GPIO4 , 对角脚 -> GND
//   (GPIO4 有内部上拉, 按下=LOW, 松开=HIGH)
// 屏幕配置由 User_Setup.h 提供（ILI9341, CS=15 DC=27 MOSI=23 SCK=18 BL=32）

#include <WiFi.h>
#include <PubSubClient.h>
#include <driver/adc.h>
#include <TFT_eSPI.h>
#include <U8g2_for_TFT_eSPI.h>

// ================= 配置区 =================
const char* WIFI_SSID     = "1";
const char* WIFI_PASSWORD = "你的WiFi密码";
const char* MQTT_BROKER   = "broker.hivemq.com";   // 2026-10-26 emqx超时, 换HiveMQ

const char* TOPIC_SIGN_TEXT  = "aiot/sign/gesture/demo";
const char* TOPIC_AUDIO_SEND = "aiot/voice/audio";
const char* TOPIC_AUDIO_TEXT = "aiot/voice/text";

#define PIN_MIC_ADC 34            // 模拟麦克风 OUT
#define MIC_CHANNEL ADC1_CHANNEL_6 // GPIO34 对应 ADC1_CH6

#define USE_BUTTON 0              // 0=自动断句模式(当前使用, 听到人声就录, 不用按键)
                                  // 1=按键说话模式(以后想用按键再改回 1)
#define PIN_BUTTON 4              // 外接按键: 一脚->GPIO4, 对角脚->GND
#define PIN_BUTTON2 0             // 板载 BOOT 键(板上丝印 BOOT/IO0), 不用接线也能按
#define BTN_DEBUG 1               // 1=每秒打印候选引脚电平(接线排查用); 调好后改 0 免得刷屏
// ==========================================

WiFiClient espClient;
PubSubClient mqtt(espClient);
TFT_eSPI tft = TFT_eSPI();
U8g2_for_TFT_eSPI u8f;

// ---- 录音参数 ----
#define SAMPLE_RATE     8000
#define MAX_SECONDS     4                              // 单句最长 4 秒(超过强制发)
#define TOTAL_SAMPLES   (SAMPLE_RATE * MAX_SECONDS)     // 32000 点 = 64KB 静态
#define CHUNK_SAMPLES   800                             // 每 0.1 秒一个判断块
#define PRE_SAMPLES     4000                            // 0.5 秒预录, 防止吃掉第一个字
#define QUIET_CHUNKS    4                               // 连续 0.4 秒无声 = 说完了
#define GAIN_SHIFT      3                               // 幅度 x8 (12位ADC -> int16)
#define MIC_DEBUG       1                               // 1=每秒打印麦克风电平(排查用); 正常后改0

// ---- 开口判定参数(用能量RMS, 不用瞬时峰值: 峰值对突发噪音太敏感, 对说话反而不灵) ----
#define VOICE_RMS_K     2.2f    // 开口: RMS 超过底噪的多少倍
#define VOICE_RMS_MIN   45.0f   // 开口绝对下限, 防底噪被压太低时乱触发
#define LOUD_CHUNKS     2       // 连续几块(每块0.1秒)超阈值才算开口 -> 短促噪音不触发
#define QUIET_RMS_K     1.35f   // 说完: RMS 回落到这个倍数以内
#define QUIET_RMS_MIN   35.0f

static int16_t pcmBuffer[TOTAL_SAMPLES];
static int16_t preBuf[PRE_SAMPLES];
static uint32_t sampleIdx = 0;
static uint32_t nextTickUs = 0;
static uint32_t voiceCount = 0;

// 屏幕上的麦克风电平(诊断用, 不用开串口就能看麦克风有没有信号)
static int   micPeakShow = 0;
static float micThShow = 0;

static bool speaking = false;
static int  quietRun = 0;
static int  loudRun = 0;              // 连续"够大声"的块数
static float noisePeak = 150.0f;      // 旧: 底噪峰值估计(保留给日志)
static float noiseRms = 30.0f;        // 底噪能量(RMS), 主判据用的
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
  }
}

// ================= 采一个 0.1 秒块 =================
// 每个块单独去直流(减块内均值), 返回块内峰值(原始ADC计数, 不含直流)
// 结果写入 dst(已放大成 int16 波形); dst 为 NULL 时只算峰值不保存
int sampleChunk(int16_t* dst) {
  static int16_t raw[CHUNK_SAMPLES];
  uint32_t now = micros();
  int32_t sum = 0;

  for (int i = 0; i < CHUNK_SAMPLES; i++) {
    while ((int32_t)(now - nextTickUs) < 0) { now = micros(); }
    int v = adc1_get_raw(MIC_CHANNEL);
    raw[i] = (int16_t)v;
    sum += v;
    nextTickUs += (1000000UL / SAMPLE_RATE);
  }

  int mean = (int)(sum / CHUNK_SAMPLES);
  int peak = 0;
  float sumSq = 0.0f;
  for (int i = 0; i < CHUNK_SAMPLES; i++) {
    int ac = raw[i] - mean;
    int a = ac < 0 ? -ac : ac;
    if (a > peak) peak = a;
    sumSq += (float)ac * (float)ac;
    if (dst) {
      int32_t v = (int32_t)ac << GAIN_SHIFT;
      if (v > 32000) v = 32000;
      if (v < -32000) v = -32000;
      dst[i] = (int16_t)v;
    }
  }
  chunkRms = sqrtf(sumSq / (float)CHUNK_SAMPLES);   // 这块的能量(人声强度)
  return peak;
}

// 上传整句
void publishUtterance() {
  if (sampleIdx < (uint32_t)(SAMPLE_RATE / 4)) {   // 短于 0.25 秒当误触发丢掉
    Serial.printf("[语音] 太短(%.2f秒), 丢弃\n", (float)sampleIdx / SAMPLE_RATE);
    sampleIdx = 0;
    speaking = false;
    quietRun = 0;
    return;
  }
  voiceCount++;
  uint32_t bytes = sampleIdx * sizeof(int16_t);
  Serial.printf("[语音 #%u] 整句 %.2f 秒 %u 字节, 底噪RMS %.0f\n",
                voiceCount, (float)sampleIdx / SAMPLE_RATE, bytes, noiseRms);
  if (mqtt.connected()) {
    bool ok = mqtt.publish(TOPIC_AUDIO_SEND, (const uint8_t*)pcmBuffer, bytes);
    Serial.printf("[MQTT] 上传 %s\n", ok ? "成功" : "失败!");
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
  }
}

// ================= Setup =================
void setup() {
  Serial.begin(115200);
  delay(200);

  // --- 麦克风 ADC 初始化 ---
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(MIC_CHANNEL, ADC_ATTEN_DB_11); // 0~3.3V 全量程
#if USE_BUTTON
  pinMode(PIN_BUTTON, INPUT_PULLUP);   // 内部上拉, 按下=LOW
  pinMode(PIN_BUTTON2, INPUT_PULLUP);  // 板载 BOOT 键
  Serial.printf("[麦克风] ADC 就绪 (GPIO34), 按键说话模式 (外接按键=GPIO%d / 板载BOOT键, 按住录/松开发)\n", PIN_BUTTON);
#else
  Serial.println("[麦克风] ADC 就绪 (GPIO34), 自动断句模式");
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

  nextTickUs = micros();
}

// ================= Loop =================
void loop() {
  ensureWiFi();
  ensureMQTT();
  mqtt.loop();

#if USE_BUTTON
  // ---- 接线自检: 每秒把几个空闲脚的电平打出来 ----
  // 按住按键不放, 看哪个脚从 1 变成 0, 那个脚就是杜邦线实际插的脚, 改 PIN_BUTTON 即可。
  // 注意: 只在没录音时打印, 录音中打印会打断采样时序, 让声音变调。
  static unsigned long lastScan = 0;
  if (!speaking && millis() - lastScan > 1000) {
    lastScan = millis();
    const int scanPins[] = {0, 2, 4, 5, 13, 14, 25, 26, 33};
    const int nScan = sizeof(scanPins) / sizeof(scanPins[0]);
    char buf[160] = {0};
    for (int i = 0; i < nScan; i++) {
      pinMode(scanPins[i], INPUT_PULLUP);
      int lv = digitalRead(scanPins[i]);
      char one[12];
      snprintf(one, sizeof(one), "%d:%d ", scanPins[i], lv);
      strncat(buf, one, sizeof(buf) - strlen(buf) - 1);
    }
#if BTN_DEBUG
    Serial.printf("[引脚扫描] %s (1=松开 0=按下)\n", buf);
#endif
  }

  // ---- 按键模式: 按住说话, 松开发送 ----
  static bool pressed = false;    // 防抖后的稳定状态(true=按下)
  static uint8_t db = 0;
  // 外接按键 或 板载 BOOT 键, 任一按下都算
  bool raw = (digitalRead(PIN_BUTTON) == LOW) || (digitalRead(PIN_BUTTON2) == LOW);

  if (raw != pressed) {
    if (++db >= 3) {              // 连续3个循环(~6ms)一致才算数, 防抖
      pressed = raw;
      db = 0;
    }
  } else {
    db = 0;
  }

  // 用电平驱动而不是边沿触发: 按住就一定在录, 松手一定发, 不会漏
  if (pressed && !speaking) {
    speaking = true;
    sampleIdx = 0;
    quietRun = 0;
    nextTickUs = micros();        // 关键: 重新对齐 8kHz 采样时钟
                                  // (不对齐的话, 待机期间时钟停在过去, 一录音就全速采样,
                                  //  电平被压缩, 上传的声音变调, 百度必识别失败)
    Serial.println("[语音] 按下, 录音中... (松开发送, 单段最长4秒)");
  } else if (!pressed && speaking) {
    Serial.println("[语音] 松开, 发送");
    publishUtterance();
  }

  if (speaking) {
    if (sampleIdx + CHUNK_SAMPLES <= TOTAL_SAMPLES) {
      sampleChunk(pcmBuffer + sampleIdx);
      sampleIdx += CHUNK_SAMPLES;
    } else {
      // 按住超过4秒: 先把这段发出去, 下一轮会自动开始录下一段
      Serial.println("[语音] 超过4秒, 分段发送");
      publishUtterance();
      nextTickUs = micros();
    }
  }
#else
  if (!speaking) {
    // 待机: 先把预录缓冲整体前移一个块, 再把新块写到末尾(保证时间顺序正确)
    memmove(preBuf, preBuf + CHUNK_SAMPLES,
            (PRE_SAMPLES - CHUNK_SAMPLES) * sizeof(int16_t));
    int peak = sampleChunk(preBuf + PRE_SAMPLES - CHUNK_SAMPLES);
    float rms = chunkRms;

    // 底噪跟随: 只在"还算安静"时更新, 速度放慢, 免得一句话把自己的门槛抬上去
    if (rms < noiseRms * 2.0f) {
      noiseRms = noiseRms * 0.95f + rms * 0.05f;
      if (noiseRms < 20.0f) noiseRms = 20.0f;
    }

    float th = noiseRms * VOICE_RMS_K;
    if (th < VOICE_RMS_MIN) th = VOICE_RMS_MIN;

    micPeakShow = (int)rms;
    micThShow = th;

#if MIC_DEBUG
    static unsigned long lastMicLog = 0;
    if (millis() - lastMicLog > 1000) {
      lastMicLog = millis();
      Serial.printf("[麦克风] RMS %.0f | 峰值 %d | 底噪 %.0f | 阈值 %.0f\n",
                    rms, peak, noiseRms, th);
    }
#endif

    // 连续 LOUD_CHUNKS 块超阈值才算开口: 关门、咳嗽这类短促噪音就骗不过去了
    if (rms > th) {
      loudRun++;
    } else {
      loudRun = 0;
    }

    if (loudRun >= LOUD_CHUNKS) {
      speaking = true;
      quietRun = 0;
      loudRun = 0;
      // 把预录的 0.5 秒搬到正式缓冲
      memcpy(pcmBuffer, preBuf, PRE_SAMPLES * sizeof(int16_t));
      sampleIdx = PRE_SAMPLES;
      Serial.printf("[语音] 检测到开口 (RMS %.0f, 底噪 %.0f)\n", rms, noiseRms);
    }
  } else {
    int peak = sampleChunk(pcmBuffer + sampleIdx);
    sampleIdx += CHUNK_SAMPLES;
    float rms = chunkRms;

    // 判定"说完": 能量回落到接近底噪
    float quietTh = noiseRms * QUIET_RMS_K;
    if (quietTh < QUIET_RMS_MIN) quietTh = QUIET_RMS_MIN;

    if (rms < quietTh) {
      quietRun++;
      noiseRms = noiseRms * 0.9f + rms * 0.1f;   // 只在安静段适应环境
      if (noiseRms < 20.0f) noiseRms = 20.0f;
    } else {
      quietRun = 0;
    }

    if (quietRun >= QUIET_CHUNKS || sampleIdx >= TOTAL_SAMPLES) {
      publishUtterance();
    }
  }
#endif

  // 语音区下方实时显示麦克风电平: 说话时左边数字必须明显变大
  // 不开串口也能一眼看出麦克风到底有没有信号
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
