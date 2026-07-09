# Universal Adaptive Fitness Coach — Project Session Log

**Date:** July 2, 2026
**Context:** Prior lab collaboration on an LLM for exercise prescription for blind/deaf users stalled on lack of a training dataset. This session scoped and built a dataset-free, Ollama-based version in a one-week window.

---

## 1. Brainstorm: 10 dataset-free directions

The core insight: sidestep the dataset problem rather than shrink it. Ten directions considered, all buildable on Ollama without a custom training set:

1. **RAG-powered exercise assistant** — no fine-tuning; ground answers in a curated vector store of public adaptive-fitness guidelines (ACSM, NCHPAD, Paralympic/USABA, deaf-fitness orgs).
2. **Agentic safety-review pipeline** — one agent drafts, a second critiques for population-specific risk, a third formats accessibly.
3. **Voice-first coach for blind users** — speech-to-text → LLM → text-to-speech, fully screen-free.
4. **Visual cueing system for deaf users** — vision model reads photos/frames of form, outputs corrective captions since verbal cues don't land.
5. **Decision-tree + LLM personalization layer** — deterministic rules engine handles safety-critical logic; LLM only personalizes phrasing.
6. **Synthetic dataset generator** — use a larger Ollama model to manufacture synthetic Q&A pairs, then LoRA fine-tune as proof of concept (attacks the original bottleneck directly).
7. **Accessible-output formatter** — LLM rewrites prescriptions into screen-reader/braille-friendly structured text.
8. **Evaluation harness** — ~30–50 hand-written scenario prompts, multiple Ollama models scored by an LLM-judge for safety/clarity/accessibility.
9. **Function-calling intake → dual-output pipeline** — one structured intake renders into both an audio script (blind) and a visual text card (deaf).
10. **Coach-in-the-loop chat that seeds a real dataset** — logged conversations become the seed dataset the original lab project needed.

## 2. What got built

The shipped app combines elements of several of the above into a single Streamlit tool, **Universal Adaptive Fitness Coach**, with two tabs:

- **Audio Coach (blind users):** Ollama drafts a 10-step verbal exercise script → Claude reviews it against an accessibility/safety checklist → Piper speaks it aloud. A toggle lets the same prompt run directly on Claude instead, to show production parity.
- **Form Checker (deaf users):** a single uploaded photo (not live video) is analyzed by a local vision model, returning 3 short text corrective captions. Same Claude-toggle pattern.

Design choices carried through the build:
- **Sequential, not simultaneous, model loading** (`model_manager.py`) — unloads whatever's resident in Ollama's VRAM before loading the next model, since fitting a draft model and a vision model in 8GB VRAM at once isn't guaranteed.
- **Claude as the judgment layer** on top of the local draft — catches vague spatial language and safety gaps a fast local pass might miss.
- **Static photo over live video** for the Form Checker — a deaf user mid-set can't watch a video stream and exercise at the same time.

## 3. Models

| Purpose | Model | Notes |
|---|---|---|
| Text draft (Audio Coach) | `llama3.1:8b-instruct-q4_K_M` | Swapped in from the original `qwen2.5:7b-instruct-q4_K_M` |
| Vision (Form Checker) | `qwen2.5vl:7b` | Unchanged from initial setup |
| Review + production-parity toggle | `claude-sonnet-5` | Used in both `claude_review.py` and `claude_vision.py` |

### Model swap verification

Ran in PowerShell:
```powershell
(Get-Content ollama_draft.py) -replace 'qwen2.5:7b-instruct-q4_K_M', 'llama3.1:8b-instruct-q4_K_M' | Set-Content ollama_draft.py
Select-String -Path ollama_draft.py -Pattern "DEFAULT_MODEL"
```
Confirmed output:
```
ollama_draft.py:17:DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"
ollama_draft.py:38: model: str = DEFAULT_MODEL,
```

## 4. Piper TTS: setup pivot

**Initial blocker:** the `piper-master` folder in the project directory was the Piper *source repo*, not a compiled/runnable program — those look similar but require a build step that wasn't worth the time mid-week.

**Decision point:** offered a choice between pushing through Piper (better voice quality, more setup) versus `pyttsx3` (one-line install, more robotic voice). Chose to push through Piper.

**Resolution:** Piper has an official `pip install` path that sidesteps the exe/zip/manual-path problem entirely:
```powershell
pip install piper-tts
python -m piper.download_voices en_US-lessac-medium
```
The `piper-master` folder is no longer needed (left in place, untouched, just unused).

**Code changes made to match the pip-based API:**
- `tts_piper.py` rewritten to use the `piper` Python package (`PiperVoice.load(...)`) instead of shelling out to a manually-downloaded `.exe`.
- `app.py` updated to match the simplified function signature.
- `pipeline.py` (standalone CLI variant) updated to drop the old exe-path parameter.
- `requirements.txt` updated to include `piper-tts`.

## 5. Voice: switched to `en_GB-cori-medium`

Confirmed exact voice identifier and updated `tts_piper.py` to default to it (`DEFAULT_VOICE_MODEL = "en_GB-cori-medium.onnx"`). Download command:
```powershell
python -m piper.download_voices en_GB-cori-medium
```
Both `en_GB-cori-medium.onnx` and `en_GB-cori-medium.onnx.json` are confirmed present in the project folder.

## 6. Current state of the repo (verified against uploaded files)

Confirmed working / consistent:
- `ollama_draft.py` — `DEFAULT_MODEL` correctly set to `llama3.1:8b-instruct-q4_K_M`.
- `tts_piper.py` — correctly rewritten around the pip-installed `PiperVoice` API, defaults to `en_GB-cori-medium.onnx`.
- `claude_review.py`, `claude_vision.py`, `model_manager.py`, `vision_form_checker.py` — internally consistent, no stale references found.

### Known issues found while double-checking (not yet fixed)

These were caught by re-reading the actual uploaded files, not assumed from the chat history — they'll break the app as currently written:

1. **`app.py` will crash on import.** It still does:
   ```python
   from tts_piper import synthesize_speech, PIPER_EXE, VOICE_MODEL
   ...
   synthesize_speech(st.session_state["audio_script"], out_path, piper_exe=PIPER_EXE, voice_model=VOICE_MODEL)
   ```
   but the rewritten `tts_piper.py` no longer defines `PIPER_EXE` or `VOICE_MODEL` (only `DEFAULT_VOICE_MODEL`), and `synthesize_speech()` no longer accepts a `piper_exe` argument. This needs to become:
   ```python
   from tts_piper import synthesize_speech, DEFAULT_VOICE_MODEL
   ...
   synthesize_speech(st.session_state["audio_script"], out_path, voice_model=DEFAULT_VOICE_MODEL)
   ```
2. **`requirements.txt` (as uploaded) doesn't list `piper-tts`** despite the plan to add it — worth confirming the version on disk actually matches what you think got saved.
3. **`README.md` Step 2 is stale.** It still tells a new setup user to run `ollama pull qwen2.5:7b-instruct-q4_K_M`, but the code now expects `llama3.1:8b-instruct-q4_K_M`.
4. **`README.md` Step 4 is stale.** It still documents the old manual `piper.exe` download + `VOICES.md` + edit-`PIPER_EXE`-path flow, which the pip-based approach replaced.
5. **`.env` was uploaded into this session** and contains what looks like a live Anthropic API key. Since it left your local machine, it's worth rotating that key in the Anthropic console and double-checking `.env` is in `.gitignore` before this goes anywhere near a repo.

## 7. Files touched this session

`ollama_draft.py`, `tts_piper.py`, `app.py`, `pipeline.py`, `requirements.txt`

---

*Log compiled from this working session. Item 6 (known issues) reflects a direct re-read of the uploaded files, not the earlier chat claims about what was fixed — worth fixing before the next test run.*
