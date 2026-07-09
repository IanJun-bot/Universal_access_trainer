# Third-party notices

This project's own source code is licensed under the MIT License (see
[`LICENSE`](LICENSE)). It depends on, and at runtime downloads, third-party
software and machine-learning models that are distributed under their own
licenses. Those components are **not** covered by this project's MIT license and
remain subject to their respective terms, listed below.

Nothing in this project bundles or redistributes these components — they are
installed via `pip`/`ollama` or loaded from a CDN at runtime. This file is an
attribution and pointer, provided as a courtesy and to make each component's
terms easy to find.

## Software libraries

| Component | Role | License |
|---|---|---|
| [Streamlit](https://github.com/streamlit/streamlit) | Web UI framework | Apache-2.0 |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Pose Landmarker, browser build via CDN) | Real-time pose tracking | Apache-2.0 |
| [OpenCV](https://github.com/opencv/opencv-python) (`opencv-python`) | Video frame sampling | Apache-2.0 |
| [Piper TTS](https://github.com/rhasspy/piper) (`piper-tts`) | Local text-to-speech | MIT |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Local speech-to-text | MIT |
| [Ollama](https://github.com/ollama/ollama) | Local model runtime | MIT |
| [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) | Claude API client | MIT |
| [NumPy](https://github.com/numpy/numpy) | Numerics | BSD-3-Clause |

## Machine-learning models

| Model | Role | License / terms |
|---|---|---|
| **Llama 3.1 8B Instruct** (via Ollama) | Script drafting | **Llama 3.1 Community License** — *not* an OSI-approved open-source license. Includes an Acceptable Use Policy and an attribution requirement, and imposes conditions on services exceeding 700M monthly active users. See the [license](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE). |
| **Qwen2.5-VL 7B** (via Ollama) | Vision form analysis | Apache-2.0 (per the model card). See the [model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct). |
| **Whisper** (weights, via faster-whisper) | Speech-to-text | MIT (OpenAI). |
| **Piper voice** `en_GB-cori-medium` | TTS voice | See the voice's model card in the [Piper voices repository](https://huggingface.co/rhasspy/piper-voices). |
| **Claude** (Anthropic API) | Review pass, direct-draft, vision | Not distributed with this project; used as a hosted service under Anthropic's [commercial terms and usage policies](https://www.anthropic.com/legal). |

> **Note on Llama 3.1:** because its license is not a standard open-source
> license, anyone reusing this project should read the Llama 3.1 Community
> License and Acceptable Use Policy directly before deploying, and consider a
> fully open-licensed drafting model (e.g. an Apache-2.0 Qwen or Mistral text
> model) if unrestricted terms are required.

## Fonts and icons

| Asset | License |
|---|---|
| [Barlow / Barlow Condensed](https://fonts.google.com/specimen/Barlow) | SIL Open Font License 1.1 |
| [Material Symbols](https://fonts.google.com/icons) | Apache-2.0 |

---

*Licenses are current to the best of the author's knowledge as of the last
update to this file; each component's own license file is authoritative. If any
entry here is inaccurate, the upstream project's stated license governs.*
