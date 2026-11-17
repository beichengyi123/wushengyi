#include <WiFi.h>
#include <PubSubClient.h>
#include <TFT_eSPI.h>
#include <U8g2_for_TFT_eSPI.h>

// ================= 配置区 =================
const char* WIFI_SSID     = "1";
const char* WIFI_PASSWORD = "你的WiFi密码";

const char* MQTT_BROKER   = "broker.emqx.io";
const int   MQTT_PORT     = 1883;
const char* MQTT_TOPIC    = "aiot/sign/gesture/demo"; // 必须与Python端一致
const char* MQTT_CLIENT_ID = "ESP32_Display_Test";

#define PIN_TFT_BL 32   // 背光引脚
// ==========================================

WiFiClient espClient;
PubSubClient mqtt(espClient);
TFT_eSPI tft = TFT_eSPI();
U8g2_for_TFT_eSPI u8f; // 中文字体对象

unsigned long lastWifiRetry = 0;
unsigned long lastMqttRetry = 0;
const unsigned long RETRY_INTERVAL = 5000;

// ================= MQTT 消息回调 =================
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String text = "";
  for (unsigned int i = 0; i < length; i++) {
    text += (char)payload[i];
  }
  text.trim();
  Serial.println("收到MQTT消息: " + text);
  if(text.length() == 0) return;

  // 1. 清空【手语翻译区】的文字区（保留标题栏）
  // 坐标: x=0, y=30, 宽=320, 高=80
  tft.fillRect(0, 30, 320, 80, TFT_BLACK);

  // 2. 设置中文字体和颜色
  u8f.setFont(u8g2_font_wqy16_t_gb2312); // 文泉驿16px中文字体
  u8f.setForegroundColor(TFT_YELLOW);    // 黄色

  // 3. 将文字画到屏幕上（居中偏左显示）
  // 注意：Y=90 是文字底部基线，确保文字落在 110 的高度之内
  u8f.drawUTF8(20, 90, text.c_str());
}

// ================= 非阻塞网络重连 =================
void ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  unsigned long now = millis();
  if (now - lastWifiRetry < RETRY_INTERVAL) return;
  lastWifiRetry = now;

  Serial.println("尝试连接 WiFi...");
  WiFi.disconnect();
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  // 这里仅调用begin，不阻塞死等，让loop轮询
}

void ensureMQTT() {
  if (WiFi.status() != WL_CONNECTED) return;
  if (mqtt.connected()) return;

  unsigned long now = millis();
  if (now - lastMqttRetry < RETRY_INTERVAL) return;
  lastMqttRetry = now;

  Serial.print("尝试连接 MQTT...");
  if (mqtt.connect(MQTT_CLIENT_ID)) {
    Serial.println("成功");
    mqtt.subscribe(MQTT_TOPIC);
  } else {
    Serial.printf("失败，状态码 = %d\n", mqtt.state());
  }
}

// ================= Setup =================
void setup() {
  Serial.begin(115200);
  delay(200);

  // 1. 屏幕初始化
  pinMode(PIN_TFT_BL, OUTPUT);
  digitalWrite(PIN_TFT_BL, HIGH); // 开背光
  
  tft.init();
  tft.setRotation(1);             // 关键！旋转为 320x240 横屏
  tft.fillScreen(TFT_BLACK);
  
  u8f.begin(tft);
  u8f.setFontMode(1);             // 透明背景模式

  // 2. 绘制静态 UI 框架
  // 【上半部】手语翻译区标题
  tft.fillRect(0, 0, 320, 30, TFT_NAVY);
  tft.setTextColor(TFT_WHITE);
  tft.setTextSize(2);
  tft.setCursor(10, 6);
  tft.print("Sign Translation");

  // 【下半部】语音转写区（预留结构）
  tft.drawFastHLine(0, 120, 320, TFT_LIGHTGREY); // 分割线
  tft.fillRect(0, 121, 320, 30, TFT_DARKGREEN);
  tft.setTextColor(TFT_WHITE);
  tft.setTextSize(2);
  tft.setCursor(10, 126);
  tft.print("Voice Transcription");

  // 初始文字提示
  u8f.setFont(u8g2_font_wqy16_t_gb2312);
  u8f.setForegroundColor(TFT_DARKGREY);
  u8f.drawUTF8(20, 90, "等待手语...");
  u8f.drawUTF8(20, 200, "等待语音...");

  // 3. 网络初始化
  WiFi.mode(WIFI_STA);
  ensureWiFi();
  
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(mqttCallback);
  mqtt.setKeepAlive(30);
  mqtt.setSocketTimeout(10);
}

// ================= Loop =================
void loop() {
  // 维持网络与 MQTT 连接（防断线崩溃）
  ensureWiFi();
  ensureMQTT();
  mqtt.loop();
  
  delay(10);
}