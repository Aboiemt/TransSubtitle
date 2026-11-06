# Third-party notices

TransSubtitle uses independently licensed third-party components. The release
includes available Python package license files in `THIRD_PARTY_LICENSES/` and
the build environment's package versions in `BUILD_DEPENDENCIES.json`.

## Included speech recognition model

The Windows v0.1.0 release includes the Whisper base model in CTranslate2 format.

- Original model: https://github.com/openai/whisper
- Conversion: https://huggingface.co/Systran/faster-whisper-base
- License: MIT. The upstream copyright and license text is included at
  `models/whisper-base/LICENSE.txt` in the portable release and
  `docs/licenses/whisper-LICENSE.txt` in the source repository.

## Optional components, not included in the release

- llama.cpp: https://github.com/ggml-org/llama.cpp (MIT).
- Sakura translation models: https://huggingface.co/SakuraLLM.
  Check the exact model card and license before downloading or using a model.
  The Sakura model described in this project's original setup uses
  CC-BY-NC-SA-4.0, which restricts commercial use.
- NVIDIA CUDA libraries: obtain them separately from an authorized NVIDIA
  distribution and follow their applicable license terms.

API providers apply their own terms and pricing. API keys and third-party
accounts are not included.
