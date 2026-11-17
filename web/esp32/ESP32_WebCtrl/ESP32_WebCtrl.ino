/* =====================================================================
   无声有译 · ESP32-32D 终端固件（Web 中控版 / HTTP 轮询）
   ---------------------------------------------------------------------
   和老固件(ESP32_Voice_Final.ino)的区别只有一处：
     老固件: 屏幕文字靠订阅 MQTT            -> 依赖公网broker, 容易超时
     本固件: 屏幕文字改为轮询局域网Flask后端 -> 全程内网, 稳定
   麦克风部分(录音 -> MQTT 发给电脑识别)原样保留, 1.py 那头不用动。

   每 2 秒做一轮: 录 2 秒声音 -> 发 MQTT(给1.py识别) -> HTTP 轮询(拿屏幕文字)
   所以屏幕更新节奏是 2 秒左右, 和录音节奏一致。

   烧录: Arduino IDE, 开发板 ESP32 Dev Module, 库: TFT_eSPI / U8g2_for_TFT_eSPI
         / PubSubClient(>=2.8, 老版本没有 setBufferSize 会编译报错)
   TFT 引脚沿用你原来的 User_Setup.h, 本文件不重复配置。
   ===================================================================== */

#include <WiFi.h>
#include <PubSubClient.h>
#include <HTTPClient.h>
#include <TFT_eSPI.h>
#include <U8g2_for_TFT_eSPI.h>

/* ==================== 现场要改的参数都在这里 ==================== */
#define WIFI_SSID      "1"                  // WiFi 名
#define WIFI_PASSWORD  "你的WiFi密码"          // WiFi 密码
#define SERVER_HOST    "10.16.197.173"     // 跑 Flask 的电脑局域网IP(cmd: ipconfig 查, 换网络会变)
#define SERVER_PORT    5000                 // Flask 端口(和 app.py 里一致)

#define MIC_GAIN       4                    // 音量增益。1.py 控制台总显示"安静"就调大(6/8);
                                            // 没说话也一直触发识别就调小(2/1)
#define RECORD_SECONDS 2                    // 每次录音时长(秒), 不建议改小, 词容易被截断
/* ================================================================ */

// ---- MQTT(音频通道, 与 1.py 保持一致, 别改) ----
const char* MQTT_BROKER   = "broker.hivemq.com";
const int   MQTT_PORT     = 1883;
const char* TOPIC_AUDIO   = "aiot/voice/audio";   // ESP32 -> 电脑: 原始PCM

// ---- 麦克风 ----
#define MIC_PIN     34            // MAX9814 输出脚(ADC1)
#define SAMPLE_HZ   8000          // 8kHz, 与 1.py/百度ASR 约定一致
#define N_SAMPLES   (SAMPLE_HZ * RECORD_SECONDS)

// ---- 网络 ----
WiFiClient   wifiClient;
PubSubClient mqtt(wifiClient);
HTTPClient   http;

// ---- 屏幕 ----
TFT_eSPI tft = TFT_eSPI();
U8g2_for_TFT_eSPI u8g2;

// ---- 终端状态(轮询得到) ----
long lastVer   = -1;      // 后端版本号, 变了才重画
int  locked    = 0;
String signText = "", voiceText = "";

// 2秒@8kHz = 16000 点 = 32000 字节, MQTT 单包要发得下
static int16_t pcmBuf[N_SAMPLES];

/* ---------------- 连 WiFi / MQTT ---------------- */
void ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.print("WiFi connect");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) { delay(400); Serial.print("."); }
  Serial.print(" OK, IP: "); Serial.println(WiFi.localIP());
  Serial.print("SSID: "); Serial.println(WIFI_SSID);
}

void ensureMQTT() {
  if (mqtt.connected()) return;
  String cid = "esp32-voice-" + String((uint32_t)ESP.getEfuseMac(), HEX);
  while (!mqtt.connect(cid.c_str())) {
    Serial.print("MQTT retry..."); delay(1500); ensureWiFi();
  }
  Serial.println("MQTT OK");
}

/* ---------------- 录 2 秒声音并发给 1.py ---------------- */
void recordAndSend() {
  ensureWiFi(); ensureMQTT();

  long next = micros();
  for (int i = 0; i < N_SAMPLES; i++) {
    while (micros() - next < 1000000L / SAMPLE_HZ) { }  // 精确 8kHz 节拍
    next += 1000000L / SAMPLE_HZ;
    int raw = analogRead(MIC_PIN);            // 12位: 0~4095
    int v = (raw - 2048) * MIC_GAIN;          // 去直流(抬到中点) + 增益
    if (v > 32767)  v = 32767;                // 防削波
    if (v < -32768) v = -32768;
    pcmBuf[i] = (int16_t)v;
  }
  // 1.py 那头: np.frombuffer(payload, int16) 直接用, 格式就是原始int16小端
  bool ok = mqtt.publish(TOPIC_AUDIO, (const uint8_t*)pcmBuf, sizeof(pcmBuf));
  Serial.println(ok ? "audio sent (2s)" : "audio send FAIL");
}

/* ---------------- HTTP 轮询: 问后端要屏幕内容 ---------------- */
void pollServer() {
  ensureWiFi();
  String url = String("http://") + SERVER_HOST + ":" + SERVER_PORT + "/api/terminal/esp";
  http.begin(url);
  http.setTimeout(1500);
  int code = http.GET();
  if (code != 200) {                       // 后端没开/电脑断网: 屏幕保持原样, 不闪不报错
    http.end();
    return;
  }
  String body = http.getString();
  http.end();

  long ver = lastVer;
  int  lk  = locked;
  String s = signText, v = voiceText;

  // 逐行解析极简文本格式(后端只回这四行, 不用JSON, 省掉解析库)
  int p = 0;
  while (p >= 0) {
    int e = body.indexOf('\n', p);
    String line = (e < 0) ? body.substring(p) : body.substring(p, e);
    line.trim();
    if (line.startsWith("VER:")) ver = line.substring(4).toInt();
    else if (line.startsWith("LOCK:")) lk = line.substring(5).toInt();
    else if (line.startsWith("S:")) s = line.substring(2);
    else if (line.startsWith("V:")) v = line.substring(2);
    if (e < 0) break;
    p = e + 1;
  }

  bool changed = (ver != lastVer) || (lk != locked);
  signText = s; voiceText = v; lastVer = ver; locked = lk;
  if (changed) drawScreen();
}

/* ---------------- 画屏幕 ---------------- */
void drawScreen() {
  tft.fillScreen(TFT_BLACK);
  if (locked) {
    u8g2.setForegroundColor(TFT_RED);
    u8g2.setFont(u8g2_font_wqy16_t_gb2312);
    u8g2.setCursor(96, 118); u8g2.print("屏幕已锁定");
    u8g2.setForegroundColor(TFT_DARKGREY);
    u8g2.setCursor(84, 148); u8g2.print("请在网页端解锁");
    return;
  }
  // 布局: 上半 手语(黄) / 下半 语音(青), 与老固件一致
  tft.drawFastHLine(0, 118, 320, 0x2104);
  u8g2.setFont(u8g2_font_wqy12_t_gb2312);
  u8g2.setForegroundColor(0x6306);                    // 暗黄: 标签
  u8g2.setCursor(8, 18);  u8g2.print("手语");
  u8g2.setCursor(8, 138); u8g2.print("语音");

  u8g2.setForegroundColor(TFT_YELLOW);
  drawWrapped(signText, 8, 44, 118 - 52);
  u8g2.setForegroundColor(TFT_CYAN);
  drawWrapped(voiceText, 8, 164, 240 - 172);
}

/* 中文按屏幕宽度折行(320宽, wqy16 一字16px, 一行18个字) */
void drawWrapped(const String& txt, int x, int y, int maxH) {
  if (txt.length() == 0 || txt == " ") return;
  u8g2.setFont(u8g2_font_wqy16_t_gb2312);
  int perLine = 18, cy = y;
  for (unsigned int i = 0; i < txt.length(); i += perLine) {
    u8g2.setCursor(x, cy);
    u8g2.print(txt.substring(i, min((unsigned int)txt.length(), i + perLine)));
    cy += 22;
    if (cy - y > maxH) break;               // 超出本区就不画了
  }
}

/* ---------------- 启动 ---------------- */
void setup() {
  Serial.begin(115200);
  analogSetAttenuation(ADC_11db);           // ADC 满量程, 音频不失真
  tft.init();
  tft.setRotation(0);                       // 与老固件一致; 若方向不对改 1/3
  u8g2.begin(tft);
  u8g2.setFont(u8g2_font_wqy16_t_gb2312);

  tft.fillScreen(TFT_BLACK);
  u8g2.setForegroundColor(TFT_WHITE);
  u8g2.setCursor(76, 110); u8g2.print("无声有译 中控");
  u8g2.setForegroundColor(TFT_DARKGREY);
  u8g2.setCursor(64, 140); u8g2.print("连接WiFi与服务器...");

  ensureWiFi();
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setBufferSize(32768);                // 一次要装下 32KB 音频, PubSubClient>=2.8
  ensureMQTT();

  pollServer();                             // 开机先拉一次屏幕内容
  drawScreen();
}

void loop() {
  static unsigned long lastPoll = 0;
  mqtt.loop();                              // 保持MQTT心跳
  if (millis() - lastPoll >= 2000) {        // 每2秒轮询一次(与录音节奏一致)
    lastPoll = millis();
    pollServer();
  }
  recordAndSend();                          // 录2秒 -> 发MQTT(期间屏幕暂不刷新)
}
