# -*- coding: utf-8 -*-
"""
=====================================================================
 无声有译 · GUI 打包脚本 (build_exe.py)
---------------------------------------------------------------------
 作用: 把 gui_app.py 打成不依赖 Python 环境的 Windows 程序
 用法: D:\\python\\python.exe build_exe.py
 产物: gui/dist/无声有译/无声有译.exe   (目录版, 双击运行, 启动快)
       想改成单文件 exe: 把 ONEFILE 改成 True (启动会慢十几秒, 每次解包)

 要点(踩过的坑):
 1) 1.py 是用 importlib 动态加载的, PyInstaller 静态分析看不到它,
    所以必须用 --add-data 把 1.py / sign_model.npz / sentence_api.py /
    voice_api.py 一起拷到 exe 旁边, 程序运行时从自己目录读。
 2) mediapipe 自带 tflite 模型和数据文件, 必须 --collect-all mediapipe,
    否则双击后报 "Unable to open file ... .tflite"。
 3) 用目录版(onedir)而不是单文件: mediapipe+opencv 体积大, 单文件每次
    启动都要解压几百 MB, 演示现场等不起。
=====================================================================
"""

import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))          # .../无声有译-项目/gui
PROJ = os.path.dirname(HERE)                              # .../无声有译-项目
ONEFILE = False                                           # True = 打成单个 exe
NAME = "WuShengYi"          # 目录名用英文(路径里不能有中文, 见下面 DIST_ROOT 说明)
EXE_CN = "无声有译.exe"      # 打包完成后把 exe 改成中文名, 方便双击认

# 产物必须放在"纯英文路径"下!
# 原因: mediapipe 的 C++ 底层用窄字符 fopen 打开自己的 .binarypb/.tflite,
#       路径里有中文就报 "The path does not exist"(实测确认过)。
#       项目目录叫"无声有译-项目", 是中文的, 所以 exe 不能留在里面。
DIST_ROOT = r"C:\Users\Administrator\Desktop\WuShengYi_GUI"

# 必须跟着 exe 走的资源(运行时要按相对路径读)
RESOURCES = [
    "1.py",              # 识别核心(动态加载)
    "sign_guard.py",     # 防误检 + 手位校验(1.py 会 import)
    "sign_model.npz",    # 模型
    "sentence_api.py",   # 1.py 会 import
    "voice_api.py",      # 1.py 会 import
    "icon.ico",          # 窗口图标(tkinter iconbitmap 用)
]


def _pids_listening(port):
    """netstat 里找出监听某端口的 PID"""
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                             encoding="mbcs", errors="ignore").stdout
    except Exception:
        return set()
    pids = set()
    for line in out.splitlines():
        if "LISTENING" in line and (":%d " % port) in line:
            parts = line.split()
            if parts and parts[-1].isdigit():
                pids.add(parts[-1])
    return pids


def stop_services():
    """打包前先停掉 5000/8080。

    坑(实测): 服务进程常驻时, 它一直持有 exe 目录下 logs\\backend.log /
    logs\\agent.log 的句柄, 整个 WuShengYi 目录就改不了名,
    os.rename 报 [WinError 5] 拒绝访问, 打包中断。
    """
    killed = []
    for port in (5000, 8080):
        for pid in _pids_listening(port):
            try:
                r = subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True,
                                   text=True, encoding="mbcs", errors="ignore")
                if r.returncode == 0:
                    killed.append((port, pid))
            except Exception:
                pass
    # 桌面端软件本体开着时, 整个产物目录都改不了名, 一并结束
    try:
        r = subprocess.run(["taskkill", "/IM", EXE_CN, "/F"], capture_output=True,
                           text=True, encoding="mbcs", errors="ignore")
        if r.returncode == 0:
            killed.append(("软件本体", "-"))
    except Exception:
        pass
    if killed:
        print("打包前已停止: %s" % ", ".join("%s(pid %s)" % k for k in killed))
        time.sleep(2)
    return killed


def _res_path(f):
    """资源文件位置: 图标等 GUI 自带资源在 gui\\ 下, 识别相关在项目根"""
    return os.path.join(HERE, f) if f in ("icon.ico",) else os.path.join(PROJ, f)


def main():
    for f in RESOURCES:
        p = _res_path(f)
        if not os.path.exists(p):
            print("缺少必要文件: %s" % p)
            return 1

    stop_services()      # 必须先停, 否则 logs 被占, 产物目录改不了名

    args = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--name", NAME,
        "--onefile" if ONEFILE else "--onedir",
        "--windowed",                     # 不弹黑窗口
        # mediapipe 的 tflite 模型/数据
        "--collect-all", "mediapipe",
        # 隐式依赖
        "--hidden-import", "PIL._tkinter_finder",
        "--hidden-import", "PIL.ImageDraw",     # 1.py 里动态导入, 扫描不到
        "--hidden-import", "PIL.ImageFont",
        "--collect-submodules", "PIL",
        "--collect-submodules", "numpy",
        "--collect-submodules", "segno",        # 手机扫码二维码(纯 python, 在 gui/_libs)
        # MQTT: 1.py 是运行时动态加载的, 静态分析扫不到它里面的 paho, 必须显式收进来
        "--collect-all", "paho",
        "--paths", os.path.join(HERE, "_libs"),
        # 图标(手语手掌图形, gui/icon.ico)
        "--icon", os.path.join(HERE, "icon.ico"),
        # 产物输出到英文目录(中文路径会让 mediapipe 打不开模型)
        "--distpath", DIST_ROOT,
        "--workpath", os.path.join(HERE, "build"),
    ]
    # 资源用 --add-data 放进去(Windows 分隔符是 ;)
    for f in RESOURCES:
        args += ["--add-data", "%s;." % _res_path(f)]

    args.append(os.path.join(HERE, "gui_app.py"))

    # 旧产物先改名(不直接删, 避免删除操作被系统安全策略拦截导致打包中断)
    # 注意: 必须先把运行中的 exe 关掉, 否则目录被占用, rename 会退化成"复制+删除"从而失败
    old_dist = os.path.join(DIST_ROOT, NAME)
    if os.path.isdir(old_dist):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        bak = os.path.join(DIST_ROOT, "%s_old_%s" % (NAME, stamp))
        try:
            os.rename(old_dist, bak)      # 同盘改名, 毫秒级
            print("旧产物已改名: %s (确认新 exe 能用后可手动删除)" % bak)
        except Exception as e:
            print("旧产物改名失败: %s" % e)
            print(">> 常见原因: exe 还开着, 或后端/智能体服务还在跑(它们占着 logs 里的日志文件)")
            print(">> 处理: 关掉程序, 双击项目里 99_停止全部服务.bat, 再重新打包")
            return 1

    print("开始打包…(首次会下载/分析依赖, 约 3~8 分钟)")
    print(" ".join(args[:8]) + " …")
    r = subprocess.run(args, cwd=HERE)
    if r.returncode != 0:
        print("打包失败, 看上面 PyInstaller 的报错")
        return r.returncode

    dist = os.path.join(DIST_ROOT, NAME)
    print("\n打包完成: %s" % dist)

    # 把资源再复制一份到产物目录(onedir 下 --add-data 已放进 _internal,
    # 但 1.py 是按 exe 所在目录找的, 所以要再放一份在 exe 旁边)
    if not ONEFILE and os.path.isdir(dist):
        for f in RESOURCES:
            shutil.copy2(_res_path(f), os.path.join(dist, f))
        # segno 二维码库: 纯 python 包, 整目录拷进 _internal,
        # 运行时 gui_app 把 _internal/_libs 插进 sys.path 后即可 import
        try:
            segno_src = os.path.join(HERE, "_libs", "segno")
            segno_dst = os.path.join(dist, "_internal", "_libs", "segno")
            if os.path.isdir(segno_src):
                shutil.copytree(segno_src, segno_dst, dirs_exist_ok=True)
                print("已附带 segno 二维码库 -> _internal/_libs/segno")
            else:
                print("警告: 找不到 %s, 手机扫码功能将不可用" % segno_src)
        except Exception as e:
            print("segno 复制失败: %s" % e)
        # 把项目根路径写成提示文件: exe 与项目分居两地, 启动时要靠它找回 web/ 和 agent/
        try:
            with open(os.path.join(dist, "root_hint.txt"), "w", encoding="utf-8") as fh:
                fh.write(PROJ)
            print("已写入 root_hint.txt -> %s" % PROJ)
        except Exception as e:
            print("root_hint.txt 写入失败: %s" % e)
        # exe 改成中文名, 双击区一眼能认出(改名不影响运行时找 _internal)
        src_exe = os.path.join(dist, NAME + ".exe")
        if os.path.exists(src_exe):
            try:
                os.replace(src_exe, os.path.join(dist, EXE_CN))
            except Exception as e:
                print("exe 改名失败(保留原名): %s" % e)
        print("已把 1.py / 模型 / 依赖模块复制到 exe 同目录")
        print("双击运行: %s" % os.path.join(dist, EXE_CN))
        print("注意: 整个 %s 目录不要挪进含中文的路径, 否则 mediapipe 读不到模型" % DIST_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
