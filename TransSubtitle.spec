# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 规格：中日双语字幕 TransSubtitle.exe"""

from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = []
binaries = []
hiddenimports = [
    "comtypes",
    "comtypes.stream",
    "comtypes.client",
    "soundcard",
    "customtkinter",
    "darkdetect",
    "faster_whisper",
    "ctranslate2",
    "av",
    "tokenizers",
    "onnxruntime",
    "huggingface_hub",
    "dotenv",
    "openai",
    "httpx",
    "anyio",
    "sniffio",
    "jiter",
    "pydantic",
    "numpy",
]

for pkg in (
    "customtkinter",
    "faster_whisper",
    "ctranslate2",
    "av",
    "onnxruntime",
    "tokenizers",
    "soundcard",
    "certifi",
):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# Whisper 显卡加速所需 cuBLAS（pip: nvidia-cublas-cu12）；运行时由 asr._setup_cuda 预加载
try:
    import glob as _glob
    import os as _os
    import nvidia.cublas as _nc

    for _d in _nc.__path__:
        for _f in _glob.glob(_os.path.join(_d, "bin", "cublas*64_12.dll")):
            binaries.append((_f, "nvidia/cublas/bin"))
except ImportError:
    pass

# customtkinter 主题资源
try:
    datas += collect_data_files("customtkinter")
except Exception:
    pass

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "scipy", "pandas", "tkinter.test"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TransSubtitle",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # 无黑框；调试可改 True
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TransSubtitle",
)
