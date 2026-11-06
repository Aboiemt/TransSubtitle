# Release validation — v0.1.0

Validated on November 6, 2026 (Asia/Taipei), Windows x64, Python 3.12.10.
Application behavior was preserved; publication changes cover documentation,
packaging and verification scripts. The release executable is unsigned, as requested.

## Results

| Check | Observed result |
| --- | --- |
| Project virtual environment | `pip check`: no broken requirements |
| Python syntax | `compileall` completed successfully |
| Source CLI help | Completed successfully |
| Source audio enumeration | 2 WASAPI loopback and 2 ordinary input devices |
| Whisper base CPU load | Passed, 3.500 seconds on this host |
| Offline speech inference | Passed; 8.985 seconds of synthetic Mandarin audio processed in 0.575 seconds |
| System playback → WASAPI → Whisper | Passed; 9.830 seconds captured and recognizable Chinese text returned |
| Sentence and simultaneous streaming API integration | Passed against a local mock OpenAI-compatible SSE endpoint; 2 requests |
| Portable EXE `--help` | Exit code 0 |
| Portable EXE `--list-devices` | Exit code 0 with a normal process launch |
| Portable GUI startup | Subtitle and API settings windows opened; subtitle accessibility tree retrieved |
| ZIP integrity | Passed, 1,936 files |
| Public configuration | Blank API key, API backend, base model |
| Publication credential scan | No matches for local API secrets in candidate source files or scanned package files |
| Private credential containers | No `.env`, `.pfx`, `.p12` or `.key` files in ZIP |

Test speech was synthesized locally with Microsoft Huihui Desktop; it was not
sent to an external API. Recognition is imperfect. The file transcription was:

> 你好,欢迎使用诗诗字幕翻译。今天天气很好,我们一起学习中文和日语。

The loopback transcription was:

> 你好,欢迎使用识识字幕翻译。今天天气很好,我们一起学习中文和日语。

The streaming API check used fixed mock responses to test request routing,
streaming parsing and callbacks. It does not measure translation quality or
prove compatibility with every external provider.

## Artifact

- File: `TransSubtitle-0.1.0-windows-x64.zip`
- Size: **248,039,363 bytes**
- SHA-256: `41a8408a196275e22cc90914a00b1ea19142095efe8a73244ac4801b54f48615`
- Includes the Whisper base CTranslate2 model, its MIT license, application
  dependencies and available dependency license notices.
- Excludes local private settings, Sakura, llama.cpp, CUDA and larger models.

## Limits and unexecuted checks

- Full visual GUI interaction was not executed: the Windows capture helper
  returned `window capture timed out: timed out waiting on channel`. Both app
  windows were enumerated and the subtitle accessibility tree was read.
  To inspect the GUI manually, run `.\release\v0.1.0\TransSubtitle\TransSubtitle.exe`.
- Redirecting stdout/stderr explicitly caused the frozen device-list command to
  time out on this host. Launching it normally succeeded. This is a windowed app
  and does not provide a console output interface. To repeat the successful check:
  `Start-Process .\release\v0.1.0\TransSubtitle\TransSubtitle.exe -ArgumentList '--list-devices' -Wait -PassThru`.
- WASAPI recording emitted data-discontinuity warnings on the host's virtual
  audio path; capture and recognition completed successfully.
- External translation-provider calls, GPU inference, local Sakura inference and
  a second clean Windows installation were **not executed** for this release.
  For an external end-to-end check, run `python main.py --model base`, configure
  your provider in API settings, play audio, and click Start. For GPU testing,
  install a compatible CUDA runtime first, then run
  `python main.py --model large-v3-turbo`. For Sakura testing, install the optional
  files described in the README, run `python main.py`, and select Local Sakura.

## Repeat the source verification

Place a CTranslate2 Whisper base model in `models/whisper-base/`. Create a
16 kHz, mono, signed 16-bit Mandarin WAV containing the word 字幕. On Windows with
Microsoft Huihui Desktop installed, the following creates the sample used here:

```powershell
Add-Type -AssemblyName System.Speech
$speech = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speech.SelectVoice('Microsoft Huihui Desktop')
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$speech.SetOutputToWaveFile((Join-Path $PWD 'speech-test.wav'), $format)
$speech.Speak('你好，欢迎使用实时字幕翻译。今天天气很好，我们一起学习中文和日语。')
$speech.Dispose()
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python.exe scripts\verify_source.py --sample speech-test.wav --output source-validation.json
```

The verification plays the sample through the default speaker and records its
loopback. Run it with no other audio playing. The API integration test uses only
a temporary loopback HTTP server with a dummy key.
