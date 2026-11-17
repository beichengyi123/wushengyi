# -*- coding: utf-8 -*-
"""
无声有译 —— 双向手语语音文字翻译终端（电脑端）
功能：MediaPipe Hands 识别手势 → 防抖 → MQTT 发布中文文本给 ESP32
依赖：opencv-python, mediapipe, paho-mqtt
运行：
    python gesture_mqtt_publisher.py            # 摄像头手势识别模式
    python gesture_mqtt_publisher.py --test     # 键盘输入测试模式（先验证 ESP32 显示）
"""

import sys
import time
import argparse

import cv2
import mediapipe as mp
import paho.mqtt.client as mqtt

# ==================== MQTT 配置（必须与 ESP32 端一致）====================
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT   = 1883
MQTT_TOPIC  = "aiot/sign/gesture/demo"        # 与 esp32 代码里的 MQTT_TOPIC 完全一致
CLIENT_ID   = "python_sign_sender_001"

# ==================== 手势 → 中文映射 ====================
GESTURE_MAP = {
    "OPEN_PALM":  "你好",
    "FIST":       "谢谢",
    "INDEX_ONLY": "再见",
    "THUMB_UP":   "是",
    "L_SHAPE":    "不是",
}

# ==================== 防抖参数 ====================
STABLE_FRAMES    = 5      # 连续 N 帧同一手势才认为稳定
PUBLISH_COOLDOWN = 1.5    # 同一手势最小发布间隔（秒）

# ==================== MediaPipe 初始化 ====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils


# ==================== MQTT 回调 ====================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] 已连接 {MQTT_BROKER}:{MQTT_PORT}")
        print(f"[MQTT] 发布主题：{MQTT_TOPIC}")
    else:
        print(f"[MQTT] 连接失败 rc={rc}（paho 会自动重连）")


def on_disconnect(client, userdata, rc):
    if rc != 0:
        print(f"[MQTT] 意外断开 rc={rc}，paho 自动重连中...")


def create_mqtt_client():
    client = mqtt.Client(client_id=CLIENT_ID, protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.reconnect_delay_set(min_delay=1, max_delay=10)
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_start()          # 后台线程自动维持连接与重连
    return client


def publish_text(client, chinese_text):
    """发布 UTF-8 中文，ESP32 端按字节接收后直接显示。"""
    info = client.publish(MQTT_TOPIC, chinese_text.encode("utf-8"), qos=0, retain=False)
    if info.rc == mqtt.MQTT_ERR_SUCCESS:
        print(f"[PUBLISH] 「{chinese_text}」 → {MQTT_TOPIC}")
    else:
        print(f"[PUBLISH] 失败 rc={info.rc}")


# ==================== 手势判定 ====================
def is_finger_extended(lm, tip_idx, pip_idx):
    """图像 y 轴向下，指尖 y 小于 PIP 关节 y 视为伸直。"""
    return lm[tip_idx].y < lm[pip_idx].y


def classify_gesture(lm):
    """输入 21 个归一化关键点，返回手势标签或 None。"""
    index_up  = is_finger_extended(lm, 8, 6)
    middle_up = is_finger_extended(lm, 12, 10)
    ring_up   = is_finger_extended(lm, 16, 14)
    pinky_up  = is_finger_extended(lm, 20, 18)
    thumb_up  = lm[4].x < lm[3].x      # 简化右手判定

    up_count = sum([index_up, middle_up, ring_up, pinky_up])

    if up_count == 4 and thumb_up:
        return "OPEN_PALM"
    if up_count == 0 and not thumb_up:
        return "FIST"
    if index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
        return "INDEX_ONLY"
    if thumb_up and up_count == 0:
        return "THUMB_UP"
    if index_up and thumb_up and not middle_up and not ring_up and not pinky_up:
        return "L_SHAPE"
    return None


# ==================== 模式1：摄像头手势识别 ====================
def run_camera_mode(client):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头，检查 USB 连接或被其他程序占用")
        return

    print("[INFO] 摄像头已打开，按 q 退出。")
    stable_counter = 0
    last_gesture = None
    last_publish_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] 读帧失败，继续...")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        current_gesture = None
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                current_gesture = classify_gesture(hand_lm.landmark)

        # 防抖
        if current_gesture is not None and current_gesture == last_gesture:
            stable_counter += 1
        else:
            stable_counter = 1 if current_gesture is not None else 0
            last_gesture = current_gesture

        now = time.time()
        if (current_gesture is not None
                and stable_counter >= STABLE_FRAMES
                and (now - last_publish_time) >= PUBLISH_COOLDOWN):
            text = GESTURE_MAP[current_gesture]
            publish_text(client, text)
            last_publish_time = now
            stable_counter = 0          # 发布后清零，防止连发

        # 画面叠加
        label = current_gesture if current_gesture else "—"
        cv2.putText(frame, f"Gesture: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Stable: {stable_counter}/{STABLE_FRAMES}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.imshow("Silent Voice - Press q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


# ==================== 模式2：键盘输入测试（先调通 ESP32 用）====================
def run_test_mode(client):
    print("[TEST] 键盘输入模式：输入要发送的中文，回车发布；输入 q 退出。")
    print("[TEST] 可用快捷词：1=你好 2=谢谢 3=再见 4=是 5=不是")
    quick = {"1": "你好", "2": "谢谢", "3": "再见", "4": "是", "5": "不是"}

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() == "q":
            break
        if not line:
            continue
        text = quick.get(line, line)
        publish_text(client, text)


# ==================== main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="键盘输入测试模式")
    args = parser.parse_args()

    client = create_mqtt_client()
    time.sleep(1)    # 给 on_connect 回调一点时间打印

    try:
        if args.test:
            run_test_mode(client)
        else:
            run_camera_mode(client)
    finally:
        print("[EXIT] 正在断开 MQTT...")
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()