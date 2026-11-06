# TransSubtitle — 中日双语实时字幕

捕获 Windows 系统播放声，使用本地 Whisper 识别语音，通过兼容 OpenAI 的 API
或可选 Sakura 本地模型翻译，以置顶字幕窗显示原文和译文。

- [下载 Windows 便携版](https://github.com/Aboiemt/TransSubtitle/releases/latest)
- [完整英文说明](README.md)
- [发布验证记录](docs/RELEASE_VALIDATION.md)

下载 ZIP 后完整解压，运行 `TransSubtitle.exe`，打开“API设置”，填写自己的
接口地址、密钥并选择模型，然后播放音频、点击“开始”。首版自带 Whisper base
模型，无需安装 Python；API 翻译需要联网和自行配置账号。程序界面为中文。

支持日译中、中译日、自动方向，同传、实时整句、5秒/10秒定时识别，以及双语、
仅原文、仅译文显示。默认使用 WASAPI 环回捕获，无需启用立体声混音。

Sakura、llama.cpp、CUDA 和更大的 Whisper 模型需要按英文文档另行安装。
本地 Sakura 负责日译中，其他方向仍走 API。请遵守具体模型的许可条件。

本版本按要求不做代码签名。请将下载文件的 SHA-256 与 Release 附带的
`SHA256SUMS.txt` 核对。不要公开本机 `.env` 和 `settings.json`。
