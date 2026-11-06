# TransSubtitle v0.1.0

The first Windows portable release of TransSubtitle: real-time Japanese–Chinese
subtitles for the audio playing on your PC.

## Highlights

- WASAPI system audio capture and local Whisper speech recognition.
- An always-on-top bilingual subtitle overlay with simultaneous, real-time and
  fixed-length capture modes.
- Streaming translation using DeepSeek or another OpenAI-compatible API.
- Optional offline Japanese-to-Chinese translation with Sakura and llama.cpp
  (runtime and model installed separately).

## Download and run

Download **TransSubtitle-0.1.0-windows-x64.zip**, extract the entire folder, and
run **TransSubtitle.exe**. Windows 10/11 x64 is required; Python is not required.
The archive includes the **Whisper base model** for CPU recognition, so no speech
model download is needed. The current app interface is in Chinese.

Open **API设置** (API settings), enter your own endpoint and API key, fetch/select
a translation model, save, play audio, and click **开始** (Start).
The template's localhost URL must be replaced unless you run that gateway.
API translation requires your own provider account and network access.

This release is **unsigned**. Sakura, llama.cpp, CUDA and larger Whisper models
are not bundled. See the English [README](https://github.com/Aboiemt/TransSubtitle#readme)
for source setup and optional components.

## Verification

The local CPU model, actual system playback → loopback capture → Whisper path,
source CLI and normal portable CLI launch passed. The application opened its
subtitle and settings windows. Streaming API integration was verified with a
local mock SSE server; external provider and GPU tests were not executed.

The ZIP contains **1,936 files** and is **248,039,363 bytes**.
Compare its SHA-256 with **SHA256SUMS.txt**:

```text
41a8408a196275e22cc90914a00b1ea19142095efe8a73244ac4801b54f48615
```

See **RELEASE_VALIDATION.md** for measured results, reproduction commands and
limitations. No personal API keys, local settings or signing keys are included.
