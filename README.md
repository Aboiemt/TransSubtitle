# TransSubtitle

**Real-time Japanese–Chinese subtitles for the audio playing on your Windows PC.**

TransSubtitle captures system audio through WASAPI loopback, recognizes speech
locally with Whisper, and displays the original text and a streaming translation
in an always-on-top subtitle overlay. Use an OpenAI-compatible translation API,
or add a local Sakura model for Japanese-to-Chinese translation.

[Download the Windows release](https://github.com/Aboiemt/TransSubtitle/releases/latest)
· [中文说明](README.zh-CN.md)
· [Release validation](docs/RELEASE_VALIDATION.md)

## Features

- System audio capture: translate audio from videos, streams, meetings or games.
- Local speech recognition with `faster-whisper`.
- Japanese → Chinese, Chinese → Japanese, and automatic translation direction.
- Simultaneous mode with incremental recognition and scrolling subtitles.
- Real-time sentence mode and fixed 5-second or 10-second audio chunks.
- Streaming translation through DeepSeek or another OpenAI-compatible endpoint.
- Optional local Sakura translation through an automatically managed llama.cpp server.
- A borderless, translucent, movable subtitle overlay with bilingual, source-only
  and translation-only display modes, plus API settings and model selection.

The application interface is currently in Chinese. This README is in English.

## Windows portable release

Requires **Windows 10/11, x64**. Python is not required for the portable release.

1. Download `TransSubtitle-0.1.0-windows-x64.zip` from
   [Releases](https://github.com/Aboiemt/TransSubtitle/releases).
2. Extract the **entire** `TransSubtitle` folder to a writable location.
3. Run `TransSubtitle.exe`. Keep `_internal` and `models` beside the executable.
4. Open **API设置** (API settings), enter your provider's API base URL and API key,
   fetch the available models, choose a translation model, and save.
5. Play some audio, select the translation direction, and click **开始** (Start).

The first release includes the **Whisper base model** and uses CPU recognition by
 default. No additional speech-model download is needed. Translation through an
API requires a working endpoint, credentials and network access. You supply your
own API account; provider charges may apply. No API keys are included.

The release does not include Sakura, llama.cpp, CUDA libraries or the larger
Whisper models. See the optional setup below. The executable is **unsigned**.

To check a downloaded archive in PowerShell:

```powershell
Get-FileHash .\TransSubtitle-0.1.0-windows-x64.zip -Algorithm SHA256
```

Compare the result with the release's `SHA256SUMS.txt`.

## Run from source

Python **3.10+** is required; the published build uses Python **3.12**.

```powershell
git clone https://github.com/Aboiemt/TransSubtitle.git
cd TransSubtitle
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item settings.example.json settings.json
.\.venv\Scripts\python.exe main.py
```

Configure the API in the settings window. Alternatively, copy `.env.example` to
`.env` and set `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL`.
Values in `settings.json` take priority over environment variables.

The default endpoint in the template is a local gateway at
`http://127.0.0.1:7863/v1`. Replace it with your actual provider's endpoint unless
you run a compatible gateway there. Select a model offered by that provider.

For a source checkout, Whisper weights are downloaded on first use unless they
already exist in `models/whisper-<name>/`. To keep downloads on another drive,
set the Hugging Face cache location **before launching**:

```powershell
$env:HF_HOME = 'G:\Program Files\codexcache\huggingface'
.\.venv\Scripts\python.exe main.py --model base
```

## Audio devices and capture modes

The default capture source follows the default Windows playback device. It uses
WASAPI loopback; enabling Stereo Mix is not normally necessary.

```powershell
# List playback loopback and microphone devices.
.\.venv\Scripts\python.exe main.py --list-devices

# Use a specific enumerated device.
.\.venv\Scripts\python.exe main.py --device 2

# Recognize Japanese and translate it into Chinese.
.\.venv\Scripts\python.exe main.py --lang ja --mode ja2zh --model base

# Simultaneous, real-time sentence, or fixed-length capture.
.\.venv\Scripts\python.exe main.py --capture simul
.\.venv\Scripts\python.exe main.py --capture realtime
.\.venv\Scripts\python.exe main.py --capture chunk5
.\.venv\Scripts\python.exe main.py --capture chunk10
```

Other direction choices are `--mode zh2ja` and `--mode auto`. Supported language
options are `--lang ja` and `--lang zh`. The same arguments work with the portable
executable, although its windowed build does not print console output.

If no subtitles appear, check that audio is playing on the selected output device,
wait for Whisper to become ready, and verify your API settings. Virtual or remote
audio devices may require selecting a physical playback loopback device explicitly.

## Optional local Sakura translation

1. Obtain a Windows llama.cpp distribution suitable for your hardware from
   [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases).
2. Put `llama-server.exe` and its required DLLs in `tools/llama/` next to the app.
3. Download a compatible Sakura GGUF model and put it in `models/`.
4. Select **本地 Sakura** (Local Sakura) in the model dropdown.

The app starts the local server on `127.0.0.1:8089` and stops the server it owns
when exiting. Local Sakura handles Japanese → Chinese; other directions still
use the configured API. Local translation is offline once its runtime and model
are installed. GPU support depends on the chosen llama.cpp build and hardware.
For CPU-only local translation, source users can set `local_gpu_layers = 0` in
`config.py`.

Review the license of the exact Sakura model you choose. The model used in the
original project setup is licensed under **CC-BY-NC-SA-4.0** and disallows
commercial use. Optional model weights are not distributed in this repository.

## Optional Whisper GPU acceleration

The source configuration can detect NVIDIA CUDA support when compatible cuBLAS
DLLs are available in `tools/cuda/`. In `auto` mode it selects `large-v3-turbo` on
supported GPU setups and falls back to `base` on CPU. Explicitly selecting
`--model base` keeps the smaller model. Additional GPU runtimes and model weights
must be installed separately; GPU execution was not validated for this release.

## Build a portable ZIP

```powershell
.\.venv\Scripts\python.exe scripts\build_release.py --version 0.1.0
```

To reproduce the model-inclusive layout, first place the CTranslate2 Whisper base
model files from [Systran/faster-whisper-base](https://huggingface.co/Systran/faster-whisper-base)
in `models/whisper-base/`, then run:

```powershell
.\.venv\Scripts\python.exe scripts\build_release.py --version 0.1.0 --include-base-model
```

Alternatively, `build.bat --version 0.1.0 --include-base-model` installs the project
requirements before building. Output goes to `release/v0.1.0/`. Existing output
is preserved: move it aside before rebuilding the same version.

The builder copies the **public settings template**, includes dependency license
notices, creates a ZIP and SHA-256 checksum, and excludes local API settings,
private keys, logs and optional local runtimes. Its default build cache is
`G:\Program Files\codexcache`; override it with `--cache-dir` on other machines.
Dependency versions used for the published release are in `requirements-release.txt`.

## Project layout

| File | Purpose |
| --- | --- |
| `main.py` | Application startup and audio/recognition/translation pipeline |
| `audio_capture.py` | WASAPI system audio capture and resampling |
| `asr.py` | Speech segmentation and local Whisper recognition |
| `simul.py` | Incremental recognition and simultaneous interpretation |
| `translator.py` | Streaming API and Sakura translation |
| `local_llm.py` | Local llama.cpp server lifecycle |
| `ui.py` | Always-on-top subtitle overlay |
| `settings_ui.py` | API configuration window |
| `settings_store.py` | Local settings persistence |
| `config.py` | Recognition, translation and UI defaults |
| `scripts/build_release.py` | Clean portable packaging and checksums |

## Privacy and third-party components

Speech recognition runs locally. API translation sends recognized text to the
endpoint you configure. API credentials are stored in plaintext in your local
`settings.json` or `.env`; keep those files private. They are excluded from Git
and are not copied into release packages.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for model attribution and
third-party licensing information. Recognition quality and latency depend on
audio quality, the selected model, hardware and API response time.
