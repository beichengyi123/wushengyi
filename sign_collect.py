# -*- coding: utf-8 -*-
# ================= 手语样本采集脚本 =================
# 用法:
#   1. 运行后摄像头打开, 画面左侧显示词表和每个词已录样本数
#   2. 按数字键选择要录的词 (1~9, 0); 按 Tab 在"第一批/第二批"之间换页
#   3. 摆好手势, 按空格开始录 1.5 秒 (45帧), 进度条走完自动保存
#   4. 按 z 撤销当前词最后一条样本; 按 q 退出
#   5. 样本自动存到 sign_data/<词名>/ 文件夹, 每条一个 .npy 文件
# 双手支持: 每帧 126 维 = 右手63维 + 左手63维 (以右手为锚点归一化,
#           保留两手相对位置); 只有一只手时另一块补零。
#           旧的单手样本(63维)训练时自动补零, 不用作废。
# 录制建议:
#   - 每遍故意变化: 手高低/左右偏一点、快慢不同、稍微换个角度
#   - 一个词录满 20 遍再换下一个词 (20 个词 = 400 遍, 分两次做完)
#   - 想加新词: 改下面 PAGE1 / PAGE2 表即可
# 第二批 5 个词是为这四句准备的:
#   你 + 叫 + 什么 + 名字        -> 你叫什么名字。
#   我 + 现在 + 要 + 去 + 学校    -> 我现在要去学校。
#   我 + 要 + 帮助 + 你          -> 我要帮助你。
#   我 + 现在 + 吃饭             -> 我现在吃饭。
# ====================================================

import cv2
import numpy as np
import mediapipe as mp
import os
import time

# ---------------- 词表 (想加词就改这里) ----------------
# 分两页: 第一批已经录完了, 第二批是要新录的 5 个词
# 按 Tab 键在两页之间切换 (画面左上角会显示当前是哪一页)
PAGE1 = {
    "1": "你好",
    "2": "谢谢",
    "3": "再见",
    "4": "我",
    "5": "你",
    "6": "吃饭",
    "7": "帮助",
    "8": "好",
    "9": "什么",
    "0": "名字",
}

PAGE2 = {
    "1": "叫",
    "2": "现在",
    "3": "要",
    "4": "去",
    "5": "学校",
}

PAGES = [
    ("第一批(已录完)", PAGE1),
    ("第二批(新录5个)", PAGE2),
]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sign_data")
RECORD_FRAMES = 45  # 1.5秒 @30fps

# ---------------- MediaPipe 初始化 ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,          # 双手模式
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

def extract_features(result):
    """双手特征提取: 返回 126 维 (右手63 + 左手63), 只有一只手时另一块补零
       锚点 = 右手 (没有右手就用左手), 两只手用同一个原点和缩放,
       这样两只手的相对位置被完整保留 -> 双手词也能认"""
    if not result.multi_hand_landmarks:
        return None
    by_label = {}
    for i, hand_lm in enumerate(result.multi_hand_landmarks):
        label = result.multi_handedness[i].classification[0].label  # "Left"/"Right"
        by_label[label] = np.array([[p.x, p.y, p.z] for p in hand_lm.landmark])
    anchor = by_label.get("Right", by_label.get("Left"))
    if anchor is None:
        return None
    origin = anchor[0]
    scale = np.linalg.norm(anchor[9] - origin)
    if scale < 1e-6:
        return None
    a = ((anchor - origin) / scale).flatten()
    other = by_label.get("Left") if "Right" in by_label else by_label.get("Right")
    b = ((other - origin) / scale).flatten() if other is not None else np.zeros(63)
    return np.concatenate([a, b]).astype(np.float32)

def word_dir(word):
    d = os.path.join(DATA_DIR, word)
    os.makedirs(d, exist_ok=True)
    return d

def count_samples(word):
    d = word_dir(word)
    return len([f for f in os.listdir(d) if f.endswith(".npy")])

def undo_last(word):
    d = word_dir(word)
    files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    if files:
        os.remove(os.path.join(d, files[-1]))
        return True
    return False

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("摄像头打开失败")
        return

    current = "1"          # 默认选第一个词
    page = 0               # 0=第一批, 1=第二批 (Tab 切换)
    recording = False
    rec_frames = []
    rec_progress = 0

    print("=== 手语样本采集 ===")
    print("数字键选词 | 空格录制 | z 撤销 | Tab 换页 | q 退出")

    while True:
        vocab = PAGES[page][1]
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        hand_ok = False
        if result.multi_hand_landmarks:
            hand_ok = True
            for hand_lm in result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            if recording:
                feats = extract_features(result)
                if feats is not None:
                    rec_frames.append(feats)
                rec_progress += 1

        # ---- 录制结束: 保存样本 ----
        if recording and rec_progress >= RECORD_FRAMES:
            recording = False
            if len(rec_frames) >= RECORD_FRAMES * 0.8:  # 允许丢少量帧
                word = vocab[current]
                n = count_samples(word) + 1
                path = os.path.join(word_dir(word), f"{word}_{n:03d}_{int(time.time())}.npy")
                np.save(path, np.array(rec_frames, dtype=np.float32))
                print(f"[保存] {word} 第 {n} 遍 ({len(rec_frames)}帧)")
            else:
                print("[丢弃] 有效帧不足, 手要一直保持在画面里")
            rec_frames = []
            rec_progress = 0

        # ---- 界面绘制 ----
        word = vocab[current]
        n = count_samples(word)

        # 左侧词表(只显示当前页)
        cv2.rectangle(frame, (0, 0), (170, 480), (30, 30, 30), -1)
        cv2.putText(frame, PAGES[page][0], (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        y = 52
        for key, w in vocab.items():
            cnt = count_samples(w)
            color = (0, 255, 255) if key == current else (200, 200, 200)
            mark = "> " if key == current else "  "
            cv2.putText(frame, f"{mark}{key}:{w} {cnt}", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            y += 26

        # 状态栏
        status = f"[{word}] 已录{n}遍"
        if recording:
            status += f"  录制中 {rec_progress}/{RECORD_FRAMES}"
            cv2.rectangle(frame, (180, 440), (180 + int(440 * rec_progress / RECORD_FRAMES), 465), (0, 200, 0), -1)
        elif not hand_ok:
            status += "  [未检测到手!]"
        cv2.putText(frame, status, (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 255) if recording else (0, 255, 0), 2)
        cv2.putText(frame, "SPACE=record  z=undo  TAB=page  q=quit", (180, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Sign Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            if not recording:
                recording = True
                rec_frames = []
                rec_progress = 0
        elif key == ord("z"):
            if undo_last(vocab[current]):
                print(f"[撤销] {vocab[current]} 删除最后一条")
        elif key == 9:                      # Tab: 换页
            page = (page + 1) % len(PAGES)
            print(f"[换页] 现在是 {PAGES[page][0]}")
        elif chr(key) in vocab:
            current = chr(key)

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== 各词样本统计 ===")
    total = 0
    for page_name, vocab in PAGES:
        print(f"-- {page_name} --")
        for w in vocab.values():
            c = count_samples(w)
            total += c
            flag = "OK" if c >= 20 else "不足20"
            print(f"  {w}: {c} 遍  [{flag}]")
    print(f"合计 {total} 条样本, 存放在 {DATA_DIR}")

if __name__ == "__main__":
    main()
