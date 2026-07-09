# Universal Adaptive Fitness Coach — Architecture & Build Guide

This document describes what the system does, how it's built, why each piece exists, and how to extend or modify it. It assumes the reader is an engineer who has never seen the project before.

---

## 1. What this is

A single-page Streamlit application with two independent coaching tools, unified by one idea: **the same underlying AI pipeline can serve either accessibility need if the output modality changes.**

- **Audio Coach** (for blind/low-vision users): type or select an exercise → get a 10-step verbal script written with zero visual assumptions → hear it spoken aloud automatically, at an adjustable pace.
- **Form Checker** (for deaf/hard-of-hearing users): upload a photo of your exercise form → get 3 short text corrections. No audio anywhere in this path — captions and visual feedback only.

Both tools have a toggle to run the same prompt through Claude (Anthropic's API) instead of the local Ollama model, so the same prompt architecture is demonstrably portable between a local/free model and a production model.

**Non-goal:** this is not a general fitness app. It doesn't track workouts, store user history, or personalize beyond a free-text "context" field. It's a demonstration of an accessibility-first prompt pipeline with a genuinely usable interface around it.

---

## 2. High-level architecture

```
                         ┌─────────────────────────────────────────┐
                         │              app.py (Streamlit)          │
                         │  - page config, CSS injection            │
                         │  - two tabs, all UI wiring                │
                         │  - session_state orchestration            │
                         └───────────────┬───────────────┬──────────┘
                                          │               │
                    ┌─────────────────────┘               └─────────────────────┐
                    ▼                                                            ▼
        ┌───────────────────────┐                                  ┌───────────────────────┐
        │   AUDIO COACH PATH     │                                  │  FORM CHECKER PATH     │
        └───────────────────────┘                                  └───────────────────────┘
                    │                                                            │
      ┌─────────────┴─────────────┐                                ┌────────────┴────────────┐
      ▼                           ▼                                ▼                          ▼
ollama_draft.py            claude_review.py                vision_form_checker.py     claude_vision.py
(local draft via           (Claude QC pass,                (local vision model,       (Claude vision API,
 llama3.1, HTTP to         or full draft-with-              qwen2.5vl, via ollama      same prompt, toggle
 localhost:11434)          Claude toggle path)              python package)           path)
      │                           │                                │                          │
      └─────────────┬─────────────┘                                └────────────┬─────────────┘
                     ▼                                                           ▼
              approved script text                                   3 text corrections
                     │
                     ▼
              tts_piper.py
              (Piper TTS, local,
               CPU-only, in-memory
               WAV bytes out)
                     │
                     ▼
              st.audio(..., autoplay=True)

        model_manager.py sits underneath both local-model paths, unloading whichever
        Ollama model isn't currently needed so the draft model and vision model never
        fight for VRAM at the same time.
```

Everything runs in one Python process started by `streamlit run app.py`. There is no separate backend/frontend split, no database, no queue. State lives in Streamlit's `session_state` (per-browser-session, in-memory, lost on server restart).

---

## 3. File-by-file reference

### `app.py` — the entire UI and orchestration layer

This is the only file Streamlit executes directly. Every user interaction (button click, checkbox toggle, text input) causes Streamlit to **re-run this entire file top to bottom** — that execution model shapes almost every design decision below. There is no persistent server-side loop; `session_state` is the only thing that survives between reruns.

**The onboarding stage machine (the app's front door).** The app no longer opens on tabs — it opens on a spoken, keyboard-first question flow, driven by `st.session_state["stage"]`:

- `q1` → "Are you blind or deaf?" — displayed as large text *and* spoken via a Piper-synthesized prompt (`_prompt_audio`, cached with `@st.cache_data`). Two buttons with **native keyboard shortcuts** (`st.button(..., shortcut="Space")` / `shortcut="D"` — supported in Streamlit ≥1.58; only "C" and "R" are reserved). Space → `q2`; D → `deaf_active`. A quiet tertiary "Skip — show all tools" goes to `all_active` (the old three-tab layout) for demos and mixed use.
- `q2` → "Would you like to learn an exercise? Press Space for yes" → `blind_active`.
- `blind_active` → the Audio Coach rendered **without tabs** (nothing to visually navigate), preceded one-shot by the spoken question "What exercise would you like to perform today?..." after which **the microphone opens automatically** and Space stops/toggles recording from anywhere on the page. The spoken request is parsed by `_parse_voice_request` ("[exercise] with [equipment] while/on [place]" → exercise field + context field, rightmost-separator matching, Whisper's stray punctuation stripped) and generation auto-triggers. Net interaction for a blind user: **two keypresses + one spoken sentence**, end to end.
- `deaf_active` → Form Checker + Live Tracker tabs only. Every mode has a "Start over" button returning to `q1`.
- **Voice retry (no silent dead ends)**: if a recording transcribes to nothing, the app *speaks* "I didn't catch that, say your exercise again now" and reopens the microphone automatically — up to twice, then stops with a visible message (if the mic is genuinely broken, an endless spoken loop is worse than stopping). A "Didn't catch it? Try again — press T" button (`shortcut="T"`) triggers the same spoken-prompt-plus-mic-reopen at any time, clearing the transcription hash guard so an identical re-recording still processes. The mic auto-open JS hooks the *currently playing* audio element (not merely the last one in the DOM — an earlier script's player may still be present), with a ~3s fallback that opens the mic anyway if autoplay was blocked.

Two mechanisms make the blind path work, and both have sharp edges worth knowing:

- **The invisible voice (`_speak` / `SPEAK_JS_TEMPLATE`)**: all self-voicing — question prompts, retry prompts, and in blind mode the script itself — plays through a **single `Audio` object created on the parent window** from a zero-height component, never through an on-page `st.audio` element. This decision came from real testing and fixes three reported problems at once: (1) visible players with download menus cluttering a voice-first UI ("the voice does the typing" — an Audio object has no DOM presence and nothing to download); (2) **mid-speech stutter**, because Streamlit re-renders the page on interactions and a re-render can re-mount a playing `<audio>` element, blipping playback — the parent-window object lives outside the render cycle and survives reruns untouched; (3) reliable mic auto-open, with `ended` attached directly to the object that's speaking (`then_open_mic=True`) instead of scanning the DOM for players. A per-utterance sequence tag (`speak_seq`) guards against double-plays; each new utterance stops the previous one. **P pauses/resumes** the voice (bound in `BLIND_KEYS_JS` alongside Space→record-toggle), since there's no visible player to click. Trade-off to know: the audio ships as a base64 data-URI inside the component HTML (a long script can be several MB inline — fine on localhost, worth revisiting for remote deployment).
- **Spoken microphone state (`HELPERS_JS`)**: from real testing — "I don't know if it's recording" — every mic state change is now announced via the browser's built-in `speechSynthesis` (deliberately a *different voice* than Piper: system voice = UI feedback, Piper voice = coach content): "Recording. Speak now." finishes *before* the record click so it never leaks into the clip; "Got it. One moment." on stop. Mic buttons are targeted **by aria-label** (`record` / `stop recording` / `clear recording`), never by position — the widget's first button changes role once a clip exists, so the old first-button click hit the wrong control on retries (a clip is now cleared first, then recording restarts). A MutationObserver watches the widget for its **visual-only error strings** ("An error has occurred...", the mic-permission notice) and speaks them aloud with what to do; a permissions-API preflight announces an outright-blocked microphone with the fix.
- **Browser autoplay policy**: audio cannot auto-play before the page's first user interaction, so Q1's spoken prompt may be silent on a truly fresh load — `_speak` handles this by retrying playback on the first keypress or click (the on-screen text and a screen reader cover the gap until then). Every prompt after the first keypress plays reliably.
- **Silence must never become a script (`voice_input.py`)**: Whisper famously *hallucinates* on silence — a user who says nothing can get back "Thank you." and, before this was fixed, the app would generate a script for a phantom exercise. Three stacked defenses now: an **energy gate** (mean absolute level below ~0.004 full-scale returns empty without calling the model — verified: pure silence and synthetic room noise both return `""`, real speech still transcribes), faster-whisper's **`vad_filter=True`** (Silero VAD strips non-speech before decoding), and a **blocklist** of the classic silence-hallucinations ("thank you," "bye," "you," …). An empty transcript then triggers the spoken retry flow rather than a silent dead end.
- Verified: the full stage machine runs exception-free under AppTest (all transitions), and in a real browser a dispatched **D keypress navigated the live page** to the deaf toolset. Synthetic Space events didn't fire in that harness (Streamlit's shortcut layer matches on properties synthetic events don't fully carry) — a real spacebar supplies them; if a physical test ever disagrees, the shortcut string is a one-word change.
- Deprecation note: `components.v1.html` warns to migrate to `st.iframe`, but `st.iframe` takes a URL/path, not inline HTML — not a drop-in replacement for the pose tracker or the blind-keys component. Pinned version works; revisit on any Streamlit upgrade.

Structure of the feature renderers, top to bottom:

1. **Imports and module-level setup** — loads `.env` from the script's own directory (`Path(__file__).parent / ".env"`, not the working directory — this matters if the app is ever launched from somewhere other than its own folder).
2. **`COMMON_EXERCISES`** — a list of `(short_button_label, full_exercise_name)` tuples. The button shows the short label; the full name is what actually gets sent to the model. This split exists because early versions used the full name as the button label and it wrapped mid-word on narrow screens.
3. **`_select_and_go(name)`** — a callback (see §5, "Streamlit callback pattern") that lets a quick-select button both fill the exercise field *and* trigger generation, in one click, with no typing.
4. **`_friendly_error(e)` / `_show_error(e)`** — translates raw exception text into plain language before showing it to the user, with the original exception available in a collapsed `st.expander` for debugging. See §6.
5. **Page config + CSS injection** — `st.set_page_config(...)` then reads `style.css` (and conditionally `style_high_contrast.css`) from disk and injects it via `st.markdown(f"<style>{...}</style>", unsafe_allow_html=True)`. This happens on *every* rerun, so CSS edits take effect on the next interaction with no server restart needed — unlike changes to imported `.py` modules (see §8).
6. **High-Contrast Mode toggle** — a checkbox in the top-right area. When checked, a second stylesheet is injected *after* the base one, redeclaring the same CSS custom properties (`--color-accent`, `--color-bg`, etc.) with a yellow-on-black WCAG AAA palette. Because CSS custom properties cascade normally, every rule already written against `var(--color-*)` picks up the override automatically — no JavaScript, no body-class toggling (that was tried and doesn't work reliably in Streamlit; see §7).
7. **Hero banner** — a hand-written HTML block (`<div class="hero">...`) with an inline SVG icon (`aria-hidden="true"` since it's decorative).
8. **Two tabs** (`st.tabs(...)`), each wrapped in `st.container(border=True)` for the card look. Tab labels use Streamlit's `:material/icon_name:` shorthand for icons — this is not an emoji, it's the Material Symbols font, resolved by Streamlit's markdown renderer.
9. **Audio Coach tab body** — quick-select row → exercise/context text inputs → Claude toggle → speech-pace slider → single "Generate & speak" button → script display → audio player. See §4 for the full request flow.
10. **Form Checker tab body** — exercise name input → source radio ("Take a photo" via `st.camera_input`, the default; "Upload a photo" via `st.file_uploader`; or "Upload a video (one rep)") → Claude toggle → "Check my form" button → image preview (photo uploads only; the camera widget previews its own capture) → result card. Photo sources return the same `UploadedFile`-style object, so the single-image analysis code is identical for both. The **video option** samples the clip into ordered frames (`video_frames.extract_frames`) and sends them as one multi-image request (`check_form_sequence` / `check_form_sequence_with_claude`), so form is assessed as a *movement* — descent path, tempo, where in the rep it breaks down — rather than one frozen pose. This is still not live tracking: a vision LLM takes seconds per request, so real-time analysis is architecturally impossible on this stack (that would be a pose-estimation project, e.g. MediaPipe, not a VLM one; considered and deliberately deferred). **Reliability finding (tested, not assumed):** on frame sequences, the local `qwen2.5vl:7b` fabricates confident corrections even when the frames contain no person at all, despite two rounds of explicit abstain instructions — while Claude on the identical frames correctly answers "cannot assess: no visible person." The UI therefore shows a caution when local-model + video is selected, steering users to the Claude toggle for video. `extract_frames` raises `ValueError` with user-facing wording (clip too long, unreadable file), which `app.py` shows verbatim rather than routing through `_friendly_error`. The camera's internal button has its own testid (`stCameraInputButton`) and needed separate CSS to match the design system and 44px touch target.

### `ollama_draft.py` — local model, first-pass script

- `DRAFT_SYSTEM_PROMPT`: the constitution for what a "verbal exercise script for a blind listener" is allowed to say. It has three parts: (a) *positive* instruction to use clock-position/distance/proprioceptive language, (b) an explicit *negative* list of banned words and phrases ("look," "watch," "see," "appears," "as shown," etc.) — this list exists because a vaguer instruction ("avoid visual language") was tested and the local model still leaked visual phrasing; smaller models follow enumerated bans more reliably than abstract principles, (c) a structural constraint (exactly 10 numbered steps, no preamble, no closing remarks).
- `generate_draft(exercise_name, extra_context, model, host)`: POSTs to Ollama's `/api/chat` endpoint directly via `requests` (not the `ollama` Python package — this module predates the switch to that package elsewhere and there was no reason to change a working call). Raises `RuntimeError` with a human-readable message if Ollama isn't reachable or the model isn't pulled.
- `DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"` — chosen for being small enough to run acceptably on consumer hardware while still following multi-part instructions reasonably well. `DRAFT_SYSTEM_PROMPT` is also imported by `claude_review.py` and reused verbatim in the direct-to-Claude path, so there is exactly one source of truth for "what makes a script accessible," regardless of which model writes the first draft.

### `claude_review.py` — the quality-control layer

This is the most important architectural decision in the whole pipeline: **the local model's draft is never trusted directly.** It always passes through a second, stricter pass before reaching TTS.

- `REVIEW_SYSTEM_PROMPT`: a 5-point checklist (spatial precision, no visual dependence, safety, pacing, structure) that Claude applies to the draft. Point 2 carries the same explicit banned-word list as the draft prompt — redundant on purpose, since catching a leak here is the last line of defense before the user hears it.
- `review_script` / `draft_with_claude`: both route through a shared `_complete_text()` helper. This is the fix for the **silent empty-script bug** — Claude's extended-thinking models can spend the entire token budget on internal reasoning before writing any output text, so `stop_reason="max_tokens"` fires with zero `text`-type content blocks, and naively joining the text blocks yields `""`. No exception is thrown; the UI just shows nothing. **This was first "fixed" by raising `max_tokens` from 1024 → 4096, but the eval harness later caught it still happening intermittently at 4096** (2/15 cases returned empty on one run, then both succeeded on immediate re-run — it's non-deterministic, not a hard budget wall). The real fix has two parts: (1) `MAX_TOKENS = 8192` — since you're billed only for tokens actually generated, a high ceiling is free on normal completions and just gives the thinking phase room so text isn't crowded out; (2) `_complete_text()` retries the whole call (up to `MAX_RETRIES + 1 = 3` attempts) whenever it gets empty text *or* a `max_tokens` stop (truncated = unusable), and raises a clear `RuntimeError` if every attempt fails — so the worst case is a visible error the UI can surface, never a silent empty script. If you see empty-output-no-error from any `claude-*` extended-thinking model elsewhere, this retry-or-raise pattern is the template; don't just chase a bigger number.
- `draft_with_claude(exercise_name, extra_context, model)`: the "production mode" path. Skips Ollama entirely and asks Claude to write the script directly, using `DRAFT_SYSTEM_PROMPT` (imported from `ollama_draft.py`, not duplicated). This exists specifically to demonstrate that swapping the local model for a hosted one is a one-line change at the call site, not a prompt rewrite.
- `REVIEW_MODEL = "claude-sonnet-5"` — used for both the review pass and the direct-draft path.

### `tts_piper.py` — text to speech

- Uses **Piper**, a local, CPU-only, ONNX-based TTS engine (`pip install piper-tts`), specifically so the audio pipeline never competes with Ollama for GPU memory.
- `synthesize_speech(text, voice_model, length_scale) -> bytes`: the only public function. Returns raw WAV bytes in memory — **no file is ever written to disk.** This was a deliberate change from an earlier version that wrote to a temp file; the file was unnecessary (Streamlit's `st.audio()` accepts raw bytes directly) and periodically caused stale-file confusion.
- **The stutter bug and its fix**: Piper's high-level `voice.synthesize_wav(text, wav_file)` API (used in the first version of this file) writes each sentence's audio chunk back-to-back with **zero gap**. For a 10-sentence script that reads as words from consecutive steps running directly into each other. The fix was to drop down to the lower-level `voice.synthesize(text, syn_config)` generator, which yields one `AudioChunk` per sentence, and manually insert `SENTENCE_SILENCE_SECONDS = 0.4` seconds of silence between chunks — mirroring exactly what Piper's own CLI does internally via its `--sentence_silence` flag (which the high-level Python API does not expose).
- **The "glitchy moan" bug and its fix (found later, by ear)**: Piper's default `SynthesisConfig` has `normalize_audio=True`, which normalizes **each sentence chunk to full scale independently**. Measured on a real script: natural chunk peaks ranged 0.124–0.318, so the default was applying anywhere from ~3× to ~8× gain per sentence — short fragments ("3.", "Hold.") and trailing breath noise got amplified most, heard as a loud garbled "moan" between sentences and per-sentence volume whiplash. The fix in `synthesize_speech`: synthesize with `normalize_audio=False` (natural relative levels), apply a 10ms linear fade at each chunk edge (the silence splices otherwise click when a chunk ends off-zero), concatenate, then normalize the **whole utterance once** to 0.85 peak. Verified by measurement: windowed peaks now vary naturally instead of pinning at 1.0, and a Whisper round-trip confirms intelligibility. If TTS ever sounds glitchy again, check loudness handling before blaming the voice model.
- **Speech rate**: `DEFAULT_LENGTH_SCALE = 1.25` (see `SynthesisConfig(length_scale=...)`). Piper's `length_scale` is a phoneme-duration multiplier; `1.0` is the voice's natural pace, higher is slower. This defaults *slower than natural* because the listener has to physically perform each instruction in real time, not just absorb information — natural conversational pace was tested and reported as too fast to follow. The app exposes this as a 3-position "Speech pace" slider (Relaxed=1.25, Natural=1.0, Brisk=0.85) rather than a fixed value, on the principle that "how fast should I talk to this person" isn't something the app should assume once and lock in — the user should control it directly, the same way real screen readers expose an adjustable rate.
- `DEFAULT_VOICE_MODEL = "en_GB-cori-medium.onnx"` — resolved relative to `Path(__file__).parent` if given as a relative path, not the process's working directory. This matters if the app is ever launched via a wrapper script or IDE run configuration with a different cwd.
- Voice model files (`en_GB-cori-medium.onnx` + matching `.onnx.json`) must sit next to this file. Download via `python -m piper.download_voices en_GB-cori-medium`.

### `vision_form_checker.py` — local vision model path

- Uses the `ollama` Python package (not raw HTTP, unlike `ollama_draft.py`) to call a vision-language model with an image attached.
- `VISION_MODEL = "qwen2.5vl:7b"` — chosen for running acceptably on consumer GPUs while handling image+text prompts.
- `FORM_CHECK_PROMPT`: asks for exactly 3 short captions, explicitly bans auditory language ("listen," "hear" — the output is read, not heard, by definition for this user), and — added after review — explicitly instructs the model to say "can't assess form" when the photo is ambiguous (bad angle, blur, cropped) rather than fabricating 3 confident-sounding corrections regardless of what it can actually see. This is a trust/safety concern specific to this use case: a deaf user acting on a wrong but confidently-worded correction is worse than the model admitting uncertainty.
- Calls `switch_to(model)` from `model_manager.py` before running, to unload the audio-draft model first (see below).

### `claude_vision.py` — Claude vision API path

- Mirrors `vision_form_checker.py`'s prompt exactly (same `FORM_CHECK_PROMPT` text, kept in sync manually — there are two copies, not a shared import, because the two functions differ enough in request shape — base64 encoding, media-type sniffing — that sharing didn't reduce real duplication).
- `_media_type_for(image_bytes)`: sniffs JPEG/PNG from file header magic bytes rather than trusting the uploaded filename extension.
- `max_tokens=2048` — bumped from an original `512` for the same reason as `claude_review.py`'s token budget (extended-thinking models need headroom beyond the visible output).

### `voice_input.py` — local speech-to-text (the voice-first entry point)

- The answer to the sharpest criticism of the original build: the Audio Coach *output* was accessible (audio) but its *input* was visual (find a tab, find a field, type). Now a blind user's primary flow is: press the record control (deliberately the **first** widget in the tab, so it's the first tab stop), say "bodyweight squat, beginner," and generation starts automatically — speak, then hear.
- Uses `faster-whisper` (CTranslate2 Whisper, `base` model, ~74MB, auto-downloaded and cached on first use) on **CPU int8**, mirroring the Piper reasoning: never compete with Ollama for GPU. Transcribing a few seconds of speech takes well under a second.
- `transcribe(audio_bytes)` passes an `initial_prompt` listing exercise vocabulary — without it, short clips mishear domain words ("squat" → "squad"). Verified with a no-microphone round-trip self-test: Piper synthesizes a phrase, Whisper transcribes it back (3/4 phrases exact on synthetic voice; real human speech into a real mic differs, and the misheard case still preserved the exercise family).
- In `app.py`: `st.audio_input` (requires Streamlit ≥ 1.40-ish; present in 1.58) returns a WAV `UploadedFile`. A **hash guard** in session_state ensures each recording is transcribed exactly once, not on every rerun while the clip sits in the widget. The handler runs *before* the exercise `st.text_input` is instantiated, which is why it's allowed to set that widget's session_state key. When a request originates from voice, the synthesized audio **prepends "Your exercise: X."** — spoken confirmation of what was understood, so a mishearing is caught by ear immediately instead of discovered mid-script.
- Note `st.audio_input`, like the camera, requires a secure origin (HTTPS or localhost) for microphone access.

### `video_frames.py` — frame sampling for movement analysis

- `extract_frames(video_bytes, frame_count=6)` writes the upload to a temp file (OpenCV can't read from memory; `delete=False` because Windows won't let cv2 open a file the process still holds), samples 6 evenly spaced frames inset from the ends, caps the long edge at 1024px, and returns JPEG bytes in chronological order. Clips over 30s are rejected with a user-facing `ValueError` ("please trim to a single repetition").
- 6 frames spans one rep (start, descent, bottom, ascent, end) without bloating the multi-image request; both `qwen2.5vl` and Claude accept it comfortably in a single message.

### `pose_tracker.py` — live skeletal tracking (browser-side, third tab)

- A self-contained HTML/JS component embedded via `st.components.v1.html`: **MediaPipe Pose Landmarker running in the browser** (`@mediapipe/tasks-vision` from CDN, `pose_landmarker_lite` model, ~6MB downloaded on first load) draws a skeleton overlay on the live webcam feed at ~30fps, computes knee angle (hip–knee–ankle, both legs, taking the more visible), drives a depth bar, and counts squat reps.
- **Real-time form coaching (not just counting).** Each rep is judged on three signals, all computed from the same landmarks and evaluated when the rep completes (priority order in `evaluateRep`): **depth** (the minimum knee angle reached — shallow if it stayed above `DEPTH_SHALLOW`), **knee valgus** (knee-width ÷ ankle-width at the bottom — "caving" below `VALGUS_RATIO`; front-view only, skipped if the four leg joints aren't all visible), and **tempo** (descent duration — "rushed" below `TEMPO_MIN_SEC`). Feedback is one message per rep: "Good rep" / "Go deeper" / "Push your knees out" / "Control the descent."
- **Serves both pathways from one feature.** Every rep count and every fault is BOTH shown as big text (deaf) AND spoken via the browser speech engine, with distinct beeps for good vs. faulted reps (blind) — the accessibility-first design becoming simply the better product. A "Speak reps and feedback out loud" toggle lets a deaf user mute the audio.
- **Hands-free set control by gesture.** You can't press a button mid-squat, least of all blind. **Cross your arms in an X → end the set** (announces "Set N complete, X reps"); **wave both hands overhead → start a new set** (increments the set, resets reps). Detection is orientation-agnostic: crossed = each wrist on the opposite side of the body midline from its own shoulder, at chest height; wave = both wrists above the shoulders with sustained horizontal travel. Each requires a `GESTURE_HOLD_MS` hold and a `GESTURE_COOLDOWN_MS` lockout after firing, to fight false triggers.
- **Why browser-side JS and not `pip install mediapipe`**: the Python package risks a protobuf version conflict with recent Streamlit (mediapipe pins older protobuf), which could break the whole app; the JS build needs zero new Python dependencies. It's also the most private path in the app — frames never leave the user's machine.
- **`CFG` thresholds are calibration starting points, not settled values.** Every form and gesture threshold lives in one `CFG` object at the top of the JS. What counts as "deep enough," "knees caving," "rushed," or a valid gesture can only be tuned against a real body on camera — this is explicitly flagged in the file as a testing task, and is exactly the kind of thing that can't be got right from an armchair. Landmarks below `CFG.VIS` (0.5) visibility are ignored rather than guessed (angle shows "--"), the same abstain-over-fabricate principle as the vision prompts, enforced in geometry.
- **Verified so far**: component loads with no JS errors, the full HUD renders (Set / Reps / Knee chips, depth bar, per-rep form line, gesture hints, sound toggle), and the iframe carries `allow="camera; microphone"`. What can only be verified with a real camera + body: the rep/form/gesture thresholds themselves.
- Rep counting and form checks are squat-specific by design; other exercises are per-exercise angle definitions and state machines, not new infrastructure. Persisting sets/reps across time (history) is deliberately not built — the gesture set-control is the seed of it.

### `model_manager.py` — VRAM sequencing

- Local GPUs are typically not large enough to hold both the text-draft model and the vision model simultaneously. `switch_to(target_model)` calls `ollama.ps()` to see what's currently loaded, and force-unloads (via a zero-content chat call with `keep_alive=0`) anything that isn't the target.
- Called at the start of both `generate_draft`'s call site (in `app.py`, before invoking it) and inside `check_form`. Not needed on the Claude-API paths since those don't touch local VRAM at all.
- Important gotcha documented in the file's own docstring: `torch.cuda.empty_cache()` or similar in your own process does **nothing** to Ollama's VRAM — Ollama is a separate server process with its own GPU memory space, reachable only over HTTP/the `ollama` package, not shared memory.

### `style.css` / `style_high_contrast.css` — the design system

Injected as a raw `<style>` block via `st.markdown(unsafe_allow_html=True)`, not a Streamlit theme config — this was necessary because Streamlit's built-in theming (`.streamlit/config.toml`) doesn't reach far enough into individual widget internals (button padding, slider thumb color, etc.) to fully restyle the app.

**Design tokens** (CSS custom properties on `:root`):
```
--color-bg: #000000            (page background)
--color-surface: #1C1C1E       (input/card backgrounds)
--color-surface-raised: #2C2C2E (hover states)
--color-border: #3A3A3C
--color-text: rgba(255,255,255,0.92)
--color-text-muted: rgba(255,255,255,0.60)
--color-accent: #007AFF        (System Blue — primary actions, focus rings)
--color-accent-dark: #0060D1   (hover state for accent)
--color-success: #34C759       (System Green — Form Checker results)
--color-error: #FF3B30         (System Red — reserved for error states)
--radius-lg: 22px / --radius-md: 14px
--font-system: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif
```
No web fonts are loaded — the native OS font stack is used deliberately, both for the "feels native" aesthetic and because it means zero font-loading network requests (a real performance win, not just an aesthetic choice).

**High-Contrast Mode** (`style_high_contrast.css`) doesn't add new selectors — it **redeclares the same `:root` custom properties** with a strict yellow-on-black palette (`--color-accent: #FFFF00`, etc.) plus three overrides to strip `backdrop-filter` and force flat black backgrounds. Because every component rule in `style.css` was written against `var(--color-*)` rather than hardcoded colors, this second stylesheet is the *entire* implementation of high-contrast mode. If you add a new component with a hardcoded color instead of a variable, it will not respect High-Contrast Mode — this is the one discipline rule to maintain going forward.

**Structural CSS gotchas** (all discovered by inspecting the live DOM, not from Streamlit's documentation, which doesn't cover internal structure):
- Card containers created via `st.container(border=True)` do not get a distinct wrapper `data-testid` in this Streamlit version. The actual selector that works is `[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]` — the border lives directly on the vertical-block div, one level under the layout wrapper.
- Primary vs. secondary buttons are distinguished via `[data-testid="stBaseButton-primary"]` / `[data-testid="stBaseButton-secondary"]`, set by passing `type="primary"` to `st.button(...)` in Python. Only the two calls that represent "the one action that matters" (Generate & speak, Check my form) use `type="primary"`; everything else (quick-select chips, etc.) is left as default secondary so there's a clear visual hierarchy.
- The slider (`st.select_slider`) ships with Streamlit's default red (`#FF4B4B`) theme color with no CSS override — it does not inherit page-level color choices automatically. Override target: `[data-baseweb="slider"] [role="slider"]` for the thumb, `[data-testid="stSliderThumbValue"] p` for the floating value label.
- `[data-testid="stHeader"]` (Streamlit's own top bar, containing the Deploy button and hamburger menu) **must keep an opaque/blurred background**, not `background: transparent`. A fully transparent header was tried first and caused page content to visually show through behind the still-solid Deploy button as the user scrolled — reported as "overlapping elements." The header needs *some* backing (solid color or `backdrop-filter: blur(...)` over a semi-opaque background) to properly occlude scrolled content.
- `[data-testid="stToolbar"]`, `[data-testid="stMainMenu"]`, `[data-testid="stMainMenuButton"]` are hidden entirely (`display: none`) — this is deliberate, not an oversight: those buttons serve the developer, not the app's users, and were sitting ahead of the actual form controls in keyboard Tab order.
- `backdrop-filter: blur(...)` was originally used on both the hero banner and card containers (real glassmorphism, ~30-40px blur radius). It was **removed** after being identified as a likely cause of perceived UI "stutter" — Streamlit re-renders the entire page on every single interaction, and recompositing multiple blurred layers on every rerun is expensive on the browser's compositor. Solid, slightly-transparent-looking colors (`#16161A`, `#131316`) replace it now. If you want the glass look back, scope `backdrop-filter` to elements that don't repaint on every interaction, or accept the performance tradeoff explicitly.
- All animation lives inside `@keyframes fadeInDown` / `fadeInUp`, applied only to entrance moments (hero on load, result cards on appearance) — not continuous or decorative motion. A `@media (prefers-reduced-motion: reduce)` block at the end of `style.css` collapses all animation/transition durations to near-zero for users who've set that OS-level preference.

---

## 4. Request flow: Audio Coach, end to end

1. User either (a) clicks a quick-select chip, or (b) types an exercise name and clicks "Generate & speak."
   - (a) triggers `_select_and_go(full_name)` as an `on_click` callback: sets `session_state["audio_exercise"]` and `session_state["auto_trigger"] = True`, *then* Streamlit reruns the whole script.
   - (b) is a direct button click; `generate_clicked` is `True` on this rerun.
2. Near the top of the Audio Coach block: `auto_triggered = st.session_state.pop("auto_trigger", False)` — this is a **one-shot flag**: reading it via `.pop()` clears it immediately, so a later, unrelated rerun (e.g. the user toggling High-Contrast Mode) won't accidentally re-trigger generation.
3. `if generate_clicked or auto_triggered:` — same code path regardless of which action fired it.
4. If the "Draft directly with Claude" checkbox is on: `draft_with_claude(...)` is called once, done.
   Otherwise: `switch_to(OLLAMA_DRAFT_MODEL)` (unload the vision model if it's resident) → `generate_draft(...)` (local model, first pass) → `review_script(...)` (Claude, QC pass). The result either way is stored in `session_state["audio_script"]`.
5. If a script was produced, `synthesize_speech(script, length_scale=...)` runs immediately in the same click — the user does not press a second button to hear it. Result bytes go into `session_state["audio_bytes"]`, and `session_state["autoplay_pending"] = True` is set (another one-shot flag).
6. Below the button logic, two independent `if` blocks render whatever is currently in `session_state`, regardless of what triggered this particular rerun:
   - `if "audio_script" in session_state:` renders the script as HTML-escaped flowing text inside `.script-card` (not a `<textarea>` — see §6).
   - `if "audio_bytes" in session_state:` pops `autoplay_pending` and renders `st.audio(bytes, autoplay=<that value>)`. On the render immediately following generation, this is `True` and the browser plays the clip with no user click on a play button. On any later, unrelated rerun, it's `False` — the player is still there and manually playable, but won't force-replay itself.

Any exception anywhere in step 4 or 5 is caught and passed to `_show_error(e)`, never left to propagate to Streamlit's default traceback UI.

---

## 5. Streamlit-specific patterns used throughout (read this before modifying `app.py`)

**The whole script reruns on every interaction.** There is no persistent in-memory state except `st.session_state`, which is a dict-like object scoped to one browser session. Any variable that needs to survive past the current rerun must be written into it explicitly.

**Callback pattern (`on_click`)**: `st.button(..., on_click=fn, args=(...))` runs `fn` *before* the script reruns, which is the only supported way to programmatically set a widget's own state (you cannot do `st.session_state["audio_exercise"] = "x"` after the `st.text_input(..., key="audio_exercise")` call that owns that key has already rendered in the same pass — it must be set by a callback on the *previous* interaction, consumed on this one). This is exactly how `_select_and_go` populates the exercise field.

**One-shot flags**: any behavior that should happen exactly once, on the very next rerun, and never again on subsequent unrelated reruns, follows this pattern:
```python
# Producer (wherever the event happens):
st.session_state["some_flag"] = True

# Consumer (wherever the effect should fire):
if st.session_state.pop("some_flag", False):
    do_the_one_time_thing()
```
Used for `auto_trigger` (quick-select → generation) and `autoplay_pending` (generation → audio autoplay). If you add a new "do X once, right after Y" feature, use this pattern rather than a plain boolean that never gets cleared — a plain boolean would keep re-firing on every future rerun.

**Two widgets, same underlying key illusion**: quick-select buttons and the "Generate & speak" button both ultimately populate `session_state["audio_script"]` / `["audio_bytes"]` through the *same* code block — they are two triggers for one code path, not two separate implementations. If you add a third way to trigger generation (e.g. a voice command), make it set a flag and fall into the same `if generate_clicked or auto_triggered or new_flag:` line rather than duplicating the generation logic.

---

## 6. Accessibility decisions (why the UI looks the way it does)

- **No `<textarea>` for generated text.** Screen readers navigate editable multi-line text fields in "forms mode," reading line-by-line rather than as flowing prose. The approved script and vision captions are rendered as plain HTML (`html.escape()`'d, then wrapped in a styled `<div>`) so they read like any other paragraph of text.
- **Single-action generation.** A blind user has no efficient way to "glance and see if a second button appeared" — every additional required interaction is a real cost, not a minor inconvenience. Generation, review, and speech synthesis all happen inside one button press (or one quick-select click).
- **Autoplay, not a play button.** `st.audio(..., autoplay=True)` fires only on the render immediately following generation (see the one-shot flag pattern above) — the user doesn't have to locate and activate the native HTML5 audio player's own play control, which would be a second, separate act of navigation.
- **Adjustable speech rate**, not a single hardcoded "correct" pace — matches how real screen readers work (user-controlled rate), rather than the app deciding once what's comfortable for everyone.
- **Decorative icons are `aria-hidden="true"`** (the hero SVG) so screen readers don't announce a meaningless "image" with no useful description.
- **Streamlit's own developer chrome (Deploy button, hamburger menu) is hidden**, removing irrelevant stops from keyboard Tab order.
- **Plain-language errors with technical detail opt-in** (`_friendly_error` / `_show_error`) — the primary alert a screen reader announces is human-readable; raw exception text is available in a collapsed expander for whoever needs to actually debug it.
- **High-Contrast Mode** is a real, separately-tested visual mode (yellow on black, 7:1+ contrast), not just a marketing checkbox — see §3's CSS section for how it's implemented via variable redeclaration.
- **The AI-safety layer matters for accessibility too**: `claude_review.py`'s existence is itself an accessibility feature, not just a "production mode" demo — a script that assumes the listener can see a mirror is actively unusable, not just lower quality, for the target user.

### Known gaps (explicitly not solved, and why)

- **Screen reader interaction has not been tested against real assistive technology** (NVDA, JAWS, VoiceOver). A `time.sleep(0.6)` delay before the audio player mounts (in `app.py`, just before `st.audio(...)`) is a *mitigation* for the risk of Piper's voice and a screen reader's own page-change announcement talking over each other — it is not a verified fix. This is the single highest-priority item for whoever picks this up next: run it against NVDA (free, Windows) before claiming this works for blind users.
- **No voice input.** The exercise name is still typed or quick-selected by mouse/keyboard/switch-access; there's no speech-to-text path. This was scoped out deliberately — it requires a mic-capture browser component (Streamlit doesn't expose microphone input natively) and a transcription step, which is a new subsystem, not a UI fix. If picked up, look at `streamlit-mic-recorder` or a custom `components.html` component for capture, and either Whisper via Ollama or a hosted STT API for transcription.
- **Vision model confidence is prompt-level only**, not a real calibrated uncertainty score. The model is instructed to say "can't assess form" when the photo is ambiguous, but nothing enforces or verifies that it actually does so reliably across many photos — no automated eval exists for this yet.

---

## 7. Things that were tried and explicitly reverted (don't redo these)

- **JS-driven body-class toggling for High-Contrast Mode** (`<script>document.body.classList.add(...)</script>` injected via `st.markdown`). Doesn't work: Streamlit's markdown renderer inserts HTML via `innerHTML`-equivalent mechanics, and `<script>` tags inserted that way never execute in the browser (a general HTML behavior, not Streamlit-specific). Use the CSS-variable-redeclaration approach instead (§3).
- **Writing synthesized audio to a temp `.wav` file on disk.** Unnecessary — `st.audio()` accepts raw bytes. Also caused user-visible confusion about "the app keeps creating a new file." `tts_piper.py`'s `synthesize_speech` now returns `bytes` directly.
- **Heavy `backdrop-filter: blur()` glassmorphism** on frequently-rerendered containers. Looked good in a static screenshot, caused real perceived jank once used interactively, because Streamlit repaints the whole page on every click.
- **A large, "Apple-standard" feature spec** (ASL avatar via WebGL/Lottie + motion-capture data, a live Web Audio frequency-domain visualizer synced to TTS output, native haptics via the Vibration API, Service Worker offline caching, swapping in a hosted neural TTS engine) was proposed mid-project and deliberately **not built**. Each of those is a genuinely large subsystem on its own, several assume a custom JS frontend that Streamlit fundamentally isn't, and attempting all of them in the available time would have traded a working, tested app for a half-finished one. If any single piece of that list becomes a real priority later, scope and build it on its own — don't reach for the original spec as a checklist.
- **A second, even larger spec** ("Project Aether: The Liquid Sensory OS" — a 3D Gaussian-splatting particle avatar with real-time ASL rendering, a physics-based spring animation engine, a Web Audio frequency orb, Web Workers for image preprocessing, and §6.3 explicitly instructing "replace `st.session_state` with SWR + Zustand") was proposed later and also **not built**, for the same reason plus a sharper one: that last point isn't a UI change, it's an instruction to leave Streamlit entirely for a React frontend — a full rewrite, not a design pass. What *was* extracted from it and kept: a slow ambient "breathing" background-position animation (90s loop, `ambientBreathe` keyframe), a spring-easing button release (`--ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1)`, applied to the button's return-to-rest `transform` transition so it overshoots slightly rather than stopping flat), a pulsing "bloom" glow on focused inputs (`focusBloom` keyframe), and a subtle glow pulse on the loading spinner (`spinnerPulse` keyframe) — all in `style.css`, all covered by the existing `prefers-reduced-motion` override. If the 3D-avatar/ASL/particle pieces ever become a real priority, that's a new project with its own frontend stack, not an addition to this one.

---

## 7b. Evaluation harnesses

Two standalone scripts exist to catch regressions before a user does:

- **`eval_harness.py`** — runs a fixed set of ~15 varied exercise prompts through the script pipeline (local draft + Claude review, or `--claude-draft` for the direct path), then uses Claude as an LLM judge against the same checklist the review prompt encodes (banned visual words, unpaired clock positions, step count, no preamble). `python eval_harness.py --limit 5` for a quick pass, no flag for the full set. Writes `eval_results.json` with full detail on every case, pass or fail.
- **`eval_vision_harness.py`** — reads real photos from an `eval_photos/` folder (not included; you supply these) and runs them through the vision pipeline, checking specifically whether photos with an ambiguity hint in the filename (`blurry`, `cropped`, `dark`, etc.) actually trigger a "can't assess form" response instead of confident-but-fabricated corrections. This is the one part of the vision pipeline that can't be validated with synthetic images — it needs real people in real (including deliberately bad) photos. `--claude` switches to the Claude vision path. Caption *quality* on the normal photos still requires a human reading them; this harness only checks the abstain behavior.

**Automated WCAG audit (axe-core).** Not a script in the repo — run by injecting `axe-core` (CDN) into the live app in a browser console and calling `axe.run(document, {runOnly: {type: 'tag', values: ['wcag2a','wcag2aa','wcag21aa']}})` per screen. Last run: **zero violations** across onboarding, Audio Coach, Form Checker, Live Tracker, and High-Contrast Mode. What it caught that hand-reasoning missed: System Blue `#007AFF` is ~4.0–4.3:1 against these dark surfaces, *just below* the AA 4.5:1 text bar — hence the `--color-accent-text` / `--color-btn-primary` contrast-safe variants in `style.css` (see the comment there; the rule is: original accent for non-text uses only). Also fixed: Streamlit's `kbd` shortcut badges needed an explicit dark chip + white text inside buttons (and an inverted override in high-contrast mode). Re-run this audit after any styling change that touches color. Known limits: axe validates markup and contrast, not the actual experience — it does not descend into the pose-tracker iframe, and it is not a substitute for NVDA/human testing.

---

## 8. Running and modifying this project

### Setup
```powershell
cd accessible_exercise_pipeline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull qwen2.5vl:7b

python -m piper.download_voices en_GB-cori-medium

# .env in this same folder:
# ANTHROPIC_API_KEY=<paste-your-key-here>

streamlit run app.py
```

### Important operational gotcha (Windows)
On this environment, `streamlit run app.py` (or even `python -m streamlit run app.py` via the venv's own interpreter) has been observed to spawn its actual worker process under the **system** Python interpreter rather than the venv's copy, due to how Windows resolves the pip-generated `streamlit.exe` launcher script combined with `multiprocessing`/subprocess re-spawn internals. Symptom: the app runs, but with whichever packages happen to be installed system-wide, which may be stale or missing entirely (`ModuleNotFoundError` for `anthropic` or `ollama`, or old code from before your last edit). Fix used here: keep the venv and system Python's `pip list` in sync (`pip install -r requirements.txt` run under **both** interpreters), so it doesn't matter which one ends up serving. If you see a bug that doesn't reproduce when calling a module's functions directly in a fresh `venv\Scripts\python.exe -c "..."` shell, suspect this before anything else — check `Get-CimInstance Win32_Process | Where-Object CommandLine -like '*streamlit*'` to see which interpreter is actually bound to the port.

### Restart requirements when editing
- Editing `app.py` or either `.css` file: just refresh the browser tab, or trigger any rerun. These are read fresh (as the main script, or via `Path.read_text()`) on every single Streamlit rerun.
- Editing any **imported** module (`ollama_draft.py`, `claude_review.py`, `tts_piper.py`, `vision_form_checker.py`, `claude_vision.py`, `model_manager.py`): Streamlit's local-file watcher usually catches this and offers a "rerun," but if you don't see your change reflected, kill and restart the `streamlit run` process — Python's module cache can outlive a save-triggered rerun.

### Extension points

- **Add a new common exercise to the quick-select row**: add a `(short_label, full_name)` tuple to `COMMON_EXERCISES` in `app.py`. No other change needed — the columns and callback wiring already iterate the list.
- **Change the local draft model**: edit `DEFAULT_MODEL` in `ollama_draft.py`, `ollama pull` the new tag. No prompt changes needed unless the new model is noticeably weaker/stronger at instruction-following, in which case revisit the banned-word list's necessity.
- **Change the Claude model used for review/vision**: edit `REVIEW_MODEL` in `claude_review.py` / `VISION_MODEL` in `claude_vision.py`. If switching to a non-extended-thinking model, you can likely lower `max_tokens` back down, but there's no harm in leaving it high.
- **Change the TTS voice**: `python -m piper.download_voices <voice-name>`, then update `DEFAULT_VOICE_MODEL` in `tts_piper.py` to `"<voice-name>.onnx"`.
- **Adjust default speech pace**: change `DEFAULT_LENGTH_SCALE` in `tts_piper.py`, and/or the three values in the `{"Relaxed": 1.25, "Natural": 1.0, "Brisk": 0.85}` mapping in `app.py`.
- **Add a new accessibility rule to the script content**: add it to *both* `DRAFT_SYSTEM_PROMPT` (`ollama_draft.py`) and the checklist in `REVIEW_SYSTEM_PROMPT` (`claude_review.py`) — the draft prompt sets the initial bar, the review prompt is the enforcement backstop. Adding it to only one leaves a gap.
- **Add a new color / restyle something**: always add a CSS custom property to `:root` in `style.css` and reference it via `var(--...)` in component rules, then add the corresponding override in `style_high_contrast.css`. A hardcoded color anywhere breaks High-Contrast Mode silently.
- **Add a third tab / feature**: follow the existing pattern — `with st.container(border=True):` for the card, one primary `type="primary"` button for the main action, session_state keys namespaced by feature (e.g. `newfeature_*`), and route any exception through `_show_error()`.
