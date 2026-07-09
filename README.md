# Universal Adaptive Fitness Coach

One Streamlit app, two accessible coaching tools:

* **Audio Coach** (blind users) — generates a 10-step verbal exercise script
with no visual references, then speaks it aloud.
* **Form Checker** (deaf users) — analyzes a single uploaded photo of
exercise form and returns short text corrections. No video stream, no
audio needed — take a photo between sets, get feedback, keep lifting.

Both tabs have a toggle to run the same prompt on Claude instead of the
local model, showing the same architecture working in production.

```
                    ┌─────────────────────┐
  Exercise name  →  │   Audio Coach tab   │
                    │  Ollama draft        │
                    │  → Claude review     │  →  Piper speaks it aloud
                    └─────────────────────┘

                    ┌─────────────────────┐
  Uploaded photo →  │  Form Checker tab   │
                    │  Ollama vision model │  →  3 text corrections
                    │  (qwen2.5vl:7b)      │
                    └─────────────────────┘
```

## Why it's built this way

* **Sequential model loading, not simultaneous.** The draft model and the
vision model are never held in GPU memory at the same time. Before
either tab runs a model, `model\_manager.py` unloads whatever else is
loaded. This avoids gambling on both models fitting in 8GB VRAM together.
* **Claude does the judgment pass** on the audio script — checking for
vague spatial language and safety issues that a fast local draft might
miss.
* **Static photo, not live video**, for the Form Checker — a deaf user
exercising can't watch a video stream and do squats at the same time.
A quick photo between sets is a realistic interaction pattern.

## Setup (Windows)

### Step 1: Python environment

```powershell
cd Documents\\accessible\_exercise\_pipeline

python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

This installs Streamlit, the Ollama Python client, the Anthropic SDK, and
supporting libraries.

### Step 2: Install Ollama and pull both models

Install Ollama from https://ollama.com if you haven't already. Then pull
the two models this app uses:

```powershell
ollama pull qwen2.5:7b-instruct-q4\_K\_M
ollama pull qwen2.5vl:7b
```

The first is the text-drafting model for the Audio Coach tab (\~4.7GB).
The second is the vision model for the Form Checker tab (\~6GB on disk;
budget more at runtime for its KV cache). Because the app loads them
sequentially rather than at the same time, you don't need to fit both in
VRAM simultaneously — just individually.

Leave Ollama running in the background (it starts automatically after
install, or run `ollama serve` manually).

### Step 3: Anthropic API key

```powershell
copy .env.example .env
```

Open `.env` and paste in your Anthropic API key (it starts with `sk-ant-`):

```
ANTHROPIC_API_KEY=<paste-your-key-here>
```

Get a key at https://console.anthropic.com if you don't have one.

### Step 4: Piper TTS (for the Audio Coach tab)

1. Download `piper.exe` from https://github.com/rhasspy/piper/releases —
grab the Windows zip and unzip it, e.g. to `C:\\piper\\`.
2. Download a voice model from
https://github.com/rhasspy/piper/blob/master/VOICES.md — start with
`en\_US-lessac-medium`. You need both the `.onnx` file and its matching
`.onnx.json`, in the same folder.
3. Open `tts\_piper.py` and update `PIPER\_EXE` and `VOICE\_MODEL` at the top
to your actual paths.

If you'd rather skip this step for now, the app still works — you just
won't be able to click "Speak it aloud" until Piper is configured. The
script generation and review still run fine without it.

### Step 5: Run the app

```powershell
streamlit run app.py
```

This opens the app in your browser, usually at `http://localhost:8501`.

## Using it

**Audio Coach tab:**

1. Type an exercise name (e.g. "bodyweight squat") and optional context
(e.g. "beginner, no equipment").
2. Click "Generate script." By default this drafts with Ollama, then
reviews with Claude. Check "Draft directly with Claude" to skip the
local draft entirely and see the production-mode path.
3. Click "Speak it aloud" once Piper is set up, to hear the approved
script read out.

**Form Checker tab:**

1. Type the exercise being performed.
2. Upload a photo (jpg/png).
3. Click "Check my form." By default this uses the local vision model.
Check "Analyze with Claude instead" to compare against Claude's vision
API on the same prompt.

## Testing individual pieces

Each module runs standalone for quick debugging, without going through
the Streamlit UI:

```powershell
python ollama\_draft.py           # tests the draft step
python claude\_review.py          # tests the review step on a sample draft
python tts\_piper.py              # tests TTS, writes test\_output.wav
python vision\_form\_checker.py    # tests vision analysis (needs test\_photo.jpg)
python model\_manager.py          # prints what's currently loaded in Ollama
```

## A note on VRAM and model switching

Because the app calls `model\_manager.switch\_to()` before every model call,
switching between tabs triggers an unload-then-load cycle rather than
holding both models resident. This means:

* You'll see a short delay (a few seconds) the first time you switch from
one tab's model to the other's, while Ollama swaps what's in VRAM.
* You will not hit an out-of-memory error from having both models loaded
at once, regardless of exact VRAM usage of either model.
* If you ever see stale results or an unclear error, run
`python model\_manager.py` to check what Ollama currently has loaded.

This trades a small amount of switching latency for reliability — worth it
for a live demo, where a VRAM crash is a much worse failure mode than a
few seconds of wait.

## Known limitations (worth being upfront about)

* This is a scripting/TTS/photo-feedback tool, not a substitute for
in-person adaptive fitness instruction. It's meant to support a trainer
or companion, not replace one.
* Local model draft/analysis quality varies; Claude's review and the
Claude-vision toggle catch a lot but aren't infallible. Spot-check output
before using it with a real person.
* The vision model analyzes a single static photo, not movement over time
— it can catch position errors but not tempo, control, or momentum
issues that only show up mid-rep.

## License

This project's own source code is released under the **MIT License** — see
[`LICENSE`](LICENSE). Use it, modify it, ship it; just keep the copyright and
license notice.

It relies on third-party libraries and machine-learning models that carry their
own licenses, which are **not** covered by the MIT license above. Most are
permissive (Apache-2.0 / MIT), but one — **Llama 3.1**, used for script drafting
— is under Meta's Llama 3.1 Community License, which is *not* a standard
open-source license and has usage restrictions. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the full list, and read
that model's terms before deploying.

