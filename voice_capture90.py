# 受控采集: 90秒, 抓到的每段存档 + 统计
import time, wave
import numpy as np
import paho.mqtt.client as mqtt

segs = []
def on_connect(c,u,f,rc,props=None):
    print(f"[MQTT] connected rc={rc}", flush=True)
    c.subscribe("aiot/voice/audio", qos=0)
def on_message(c,u,msg):
    if len(msg.payload) < 3200: return
    pcm = np.frombuffer(msg.payload[:len(msg.payload)//2*2], dtype=np.int16)
    segs.append(pcm)
    print(f"  [段{len(segs)}] {len(pcm)/8000:.1f}s 峰值{int(np.abs(pcm.astype(np.float32)).max())}", flush=True)

cl = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2 if hasattr(mqtt,"CallbackAPIVersion") else None)
cl.on_connect, cl.on_message = on_connect, on_message
cl.connect("broker.hivemq.com",1883,60); cl.loop_start()
t0=time.time()
while time.time()-t0 < 90 and len(segs) < 12:
    time.sleep(0.5)
cl.loop_stop()
if segs:
    allx=np.concatenate(segs)
    with wave.open("voice_close.wav","wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(allx.astype(np.int16).tobytes())
    print(f"共{len(segs)}段 已存 voice_close.wav", flush=True)
else:
    print("没收到任何包", flush=True)
