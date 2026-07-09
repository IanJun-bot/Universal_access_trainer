"""
app.py

Universal Adaptive Fitness Coach -- an accessibility-first coaching app
with a guided, keyboard-and-voice onboarding flow:

  On load, the app asks (out loud and on screen): "Are you blind or deaf?"
    - SPACE  -> blind path: one more spoken yes/no question, then straight
                into a voice-first Audio Coach -- the exercise question is
                spoken aloud, the microphone opens automatically, and the
                user just says what they want ("squats with dumbbells while
                standing"). No typing, no visual scanning, two keypresses
                plus speech end to end.
    - D      -> deaf path: the visual tools (Form Checker + Live Tracker),
                all feedback on screen, no audio anywhere.
    - Skip   -> everything at once (for demos, testing, and mixed use).

  Audio Coach: 10-step verbal script (Ollama draft -> Claude review, or
  direct Claude), spoken aloud by Piper at an adjustable pace.
  Form Checker: photo or one-rep video -> short text corrections.
  Live Tracker: browser-side pose skeleton, knee angle, and rep counting.

Run with:
    streamlit run app.py
"""

import base64
import hashlib
import html
import time
from contextlib import nullcontext
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit.components.v1 import html as st_html

from ollama_draft import generate_draft, DEFAULT_MODEL as OLLAMA_DRAFT_MODEL
from claude_review import review_script, draft_with_claude
from tts_piper import synthesize_speech, DEFAULT_VOICE_MODEL
from vision_form_checker import check_form, check_form_sequence, VISION_MODEL
from claude_vision import check_form_with_claude, check_form_sequence_with_claude
from video_frames import extract_frames
from model_manager import switch_to
from pose_tracker import POSE_TRACKER_HTML, POSE_TRACKER_HEIGHT

load_dotenv(Path(__file__).parent / ".env")

COMMON_EXERCISES = [
    ("Squat", "Bodyweight squat"),
    ("Push-up", "Wall push-up"),
    ("Shoulder", "Seated shoulder press"),
    ("Calf raise", "Standing calf raise"),
    ("Lunge", "Chair-assisted lunge"),
]

# Spoken onboarding prompts. Q1 may be silent on a truly fresh page load
# (browsers block un-gestured audio); the text is always on screen, a
# screen reader will read it, and every prompt after the first keypress
# plays reliably because the keypress counts as the user gesture.
Q1_TEXT = (
    "Welcome to the Universal Adaptive Fitness Coach. Are you blind, or deaf? "
    "Press the space bar if you are blind. Press the letter D if you are deaf."
)
Q2_TEXT = "Would you like to learn an exercise? Press the space bar for yes."
Q3_TEXT = (
    "What exercise would you like to perform today? When you are ready, press "
    "the space bar to start recording. You will hear a short beep. Then say "
    "your exercise -- for example: squats, with dumbbells, while standing -- "
    "and press the space bar again to stop."
)
RETRY_TEXT = (
    "I didn't catch that. Press the space bar to start recording, say your "
    "exercise after the beep, then press the space bar again to stop."
)
# Spoken when the recording came back essentially silent -- a different
# problem (mic off / muted / wrong device) than "heard you but didn't
# understand," and a blind user needs to be told which.
NO_AUDIO_TEXT = (
    "I'm not hearing any sound from your microphone. It may be muted, or your "
    "browser may be using a different microphone. Check that, then press the "
    "space bar to record again."
)

# --- The invisible voice ---------------------------------------------------
# All self-voicing (question prompts, retry prompts, and in blind mode the
# script itself) plays through a SINGLE Audio object created on the PARENT
# window from an iframe component -- never through an on-page <audio>
# element. Three problems this solves at once, all reported from real use:
#   1. Visible players (with download menus) cluttering a voice-first UI --
#      an Audio object has no DOM presence at all.
#   2. Mid-speech stutter: Streamlit re-renders the page on every
#      interaction, and a re-render can re-mount a playing <audio> element,
#      restarting/blipping playback. The parent-window object lives outside
#      the render cycle entirely and survives reruns untouched.
#   3. Reliable mic auto-open: 'ended' is attached directly to the object
#      that is actually speaking, instead of scanning the DOM for players.
# The component iframe's sandbox includes allow-same-origin, so its script
# can reach the parent window -- the same mechanism used below for keys.
# Shared microphone/speech helpers, installed once on the parent window and
# used by both the speak component and the keyboard component. Design rules
# encoded here, all from real blind-flow testing:
#   - Recording must START SYNCHRONOUSLY inside the user's keypress. The
#     browser blocks getUserMedia outside a fresh user gesture, so the old
#     approach (announce "Recording, speak now", then click record from the
#     announcement's async callback) silently failed to open the mic at all.
#     A short BEEP -- not speech -- now confirms recording started, so there
#     is no announcement to delay the click and nothing to leak into the clip.
#   - Buttons are targeted by aria-label, never by position: the widget's
#     action button cycles Record/Stop recording, and a separate "Clear
#     recording" button appears once a clip exists -- clicking "the first
#     button" (the old approach) hits the wrong control on retries.
HELPERS_JS = """
  const W = window.parent;
  const D = W.document;
  // Screen-reader mode: silence the app's own speech (Piper AND browser TTS)
  // and route status/feedback to an ARIA live region instead, so the user's
  // screen reader is the only voice. Re-evaluated on every render.
  W.__uafcSilent = __SILENT__;
  if(!W.__uafcLive){
    var lr = D.createElement('div');
    lr.id = '__uafcLive'; lr.setAttribute('role','status'); lr.setAttribute('aria-live','assertive');
    lr.style.cssText = 'position:absolute;left:-9999px;top:auto;width:1px;height:1px;overflow:hidden;';
    D.body.appendChild(lr); W.__uafcLive = lr;
  }
  if(!W.__uafc){
    W.__uafc = {
      announce: function(text){
        try{ W.__uafcLive.textContent = ''; setTimeout(function(){ W.__uafcLive.textContent = text; }, 60); }catch(err){}
      },
      say: function(text, cb){
        // In screen-reader mode, hand the text to the screen reader via the
        // live region instead of speaking it ourselves.
        if(W.__uafcSilent){ W.__uafc.announce(text); if(cb) cb(); return; }
        try{
          var u = new W.SpeechSynthesisUtterance(text);
          u.rate = 1.05;
          if(cb){ u.onend = function(){ cb(); }; }
          W.speechSynthesis.cancel();
          W.speechSynthesis.speak(u);
        }catch(err){ if(cb) cb(); }
      },
      beep: function(freq, dur){
        try{
          var Ctx = W.AudioContext || W.webkitAudioContext;
          W.__uafcCtx = W.__uafcCtx || new Ctx();
          var ctx = W.__uafcCtx, t = ctx.currentTime, d = dur || 0.12;
          var o = ctx.createOscillator(), g = ctx.createGain();
          o.type = 'sine'; o.frequency.value = freq || 880;
          g.gain.setValueAtTime(0.0001, t);
          g.gain.exponentialRampToValueAtTime(0.2, t + 0.01);
          g.gain.exponentialRampToValueAtTime(0.0001, t + d);
          o.connect(g); g.connect(ctx.destination);
          o.start(t); o.stop(t + d + 0.02);
        }catch(err){}
      },
      btn: function(label){
        var c = D.querySelector('[data-testid="stAudioInput"]');
        if(!c) return null;
        var btns = c.querySelectorAll('button');
        for(var i = 0; i < btns.length; i++){
          var a = (btns[i].getAttribute('aria-label') || '').toLowerCase();
          if(a === label) return btns[i];
        }
        return null;
      },
      start: function(){
        var u = W.__uafc;
        if(u.btn('stop recording')) return;  // already recording
        var clear = u.btn('clear recording');
        if(clear){
          // A previous clip is loaded: clear it, then start fresh. Mic
          // permission is already granted by this point, so the deferred
          // restart still opens the mic even a beat after the keypress.
          clear.click();
          setTimeout(function(){ W.__uafc.start(); }, 500);
          return;
        }
        var rec = u.btn('record');
        if(rec){
          rec.click();          // synchronous, inside the gesture -> mic opens
          u.beep(880, 0.12);    // short earcon: "recording started"
        }
      },
      stop: function(){
        var s = W.__uafc.btn('stop recording');
        if(s){ s.click(); W.__uafc.beep(520, 0.12); W.__uafc.say('Got it. One moment.'); }
      },
      toggle: function(){
        if(W.__uafc.btn('stop recording')) W.__uafc.stop();
        else W.__uafc.start();
      }
    };
    // The widget reports problems only visually ("An error has occurred...",
    // the microphone-permission notice). Watch for them and say them aloud.
    try{
      var lastSaid = 0;
      var mo = new W.MutationObserver(function(){
        var c = D.querySelector('[data-testid="stAudioInput"]');
        if(!c) return;
        var t = c.textContent || '';
        var now = Date.now();
        if(now - lastSaid < 6000) return;
        if(t.indexOf('error has occurred') !== -1){
          lastSaid = now;
          W.__uafc.say('The microphone hit a problem. Press T to try again.');
        } else if(t.indexOf('would like to use your microphone') !== -1){
          lastSaid = now;
          W.__uafc.say('Your browser is asking for microphone permission. Please choose allow.');
        }
      });
      mo.observe(D.body, { subtree: true, childList: true, characterData: true });
    }catch(err){}
    // If the microphone is outright blocked, say so up front with the fix.
    try{
      if(W.navigator.permissions && W.navigator.permissions.query){
        W.navigator.permissions.query({ name: 'microphone' }).then(function(s){
          if(s.state === 'denied'){
            W.__uafc.say('Your microphone is blocked in the browser. Click the lock icon in the address bar and allow the microphone.');
          }
        }).catch(function(){});
      }
    }catch(err){}
  }
"""

SPEAK_JS_TEMPLATE = """
<script>
(function(){
__HELPERS__
  if(W.__uafcSpeakTag === "__TAG__") return;
  W.__uafcSpeakTag = "__TAG__";
  try{ if(W.__uafcPlayer){ W.__uafcPlayer.pause(); } }catch(err){}
  const a = new W.Audio("data:audio/wav;base64,__B64__");
  W.__uafcPlayer = a;
  a.addEventListener('ended', function(){ __ON_ENDED__ }, { once: true });
  a.play().catch(function(){
    // Autoplay blocked (fresh page, no user gesture yet): speak on the
    // first keypress or click instead of never.
    var unlock = function(){ a.play().catch(function(){}); };
    W.document.addEventListener('keydown', unlock, { once: true });
    W.document.addEventListener('pointerdown', unlock, { once: true });
  });
})();
</script>
"""

OPEN_MIC_JS = "W.__uafc.start();"


def _sr_mode() -> bool:
    return bool(st.session_state.get("sr_mode"))


def _helpers_js() -> str:
    """HELPERS_JS with the screen-reader silent flag filled in for this render."""
    return HELPERS_JS.replace("__SILENT__", "true" if _sr_mode() else "false")


def _live_attrs() -> str:
    """ARIA live-region attributes for a content card, only in screen-reader
    mode -- so NVDA/JAWS/VoiceOver announces prompts and results as they
    appear, taking over the voicing job the app itself does otherwise."""
    return 'role="status" aria-live="polite"' if _sr_mode() else ""


def _speak(text: str | None = None, *, wav: bytes | None = None, then_open_mic: bool = False) -> None:
    """Speak text (or pre-synthesized WAV bytes) through the parent-window
    player. Renders a zero-height component; nothing visible appears.

    In screen-reader mode this is a no-op: the app must NOT self-voice, or it
    talks over the user's screen reader. The on-screen cards carry ARIA live
    regions (see _live_attrs) so the screen reader announces the same content."""
    if _sr_mode():
        return
    audio = wav if wav is not None else _prompt_audio(text)
    seq = st.session_state.get("speak_seq", 0) + 1
    st.session_state["speak_seq"] = seq
    js = (
        SPEAK_JS_TEMPLATE
        .replace("__HELPERS__", _helpers_js())
        .replace("__TAG__", f"utt-{seq}")
        .replace("__B64__", base64.b64encode(audio).decode("ascii"))
        .replace("__ON_ENDED__", OPEN_MIC_JS if then_open_mic else "")
    )
    st_html(js, height=0)


# Keyboard helper for the blind path, bound once on the parent document:
#   Space -> toggle recording (with spoken "Recording..."/"Got it" feedback)
#   P     -> pause/resume whatever the voice is currently saying (there is
#            no visible player to click, so this is the playback control)
BLIND_KEYS_JS = """
<script>
(function(){
__HELPERS__
  if(!W.__uafcSpaceBound){
    W.__uafcSpaceBound = true;
    D.addEventListener('keydown', function(e){
      const tag = (D.activeElement && D.activeElement.tagName) || '';
      if(tag === 'INPUT' || tag === 'TEXTAREA') return;
      if(e.code === 'Space' || e.key === ' '){
        if(D.querySelector('[data-testid="stAudioInput"]')){
          e.preventDefault(); e.stopPropagation();
          W.__uafc.toggle();
        }
      } else if(e.key === 'p' || e.key === 'P'){
        const a = W.__uafcPlayer;
        if(a){ e.preventDefault(); if(a.paused){ a.play().catch(function(){}); } else { a.pause(); } }
      }
    }, true);
  }
})();
</script>
"""


def _blind_keys_component() -> None:
    st_html(BLIND_KEYS_JS.replace("__HELPERS__", _helpers_js()), height=0)


@st.cache_data(show_spinner=False)
def _prompt_audio(text: str) -> bytes:
    """Synthesize (and cache) a spoken onboarding prompt."""
    return synthesize_speech(text, voice_model=DEFAULT_VOICE_MODEL, length_scale=1.1)


def _parse_voice_request(text: str) -> tuple[str, str]:
    """
    Split a spoken request of the form
    "[exercise] with [equipment] while/on [place]" into an exercise name
    and a context string for the prompt pipeline.

    "squats with dumbbells while standing"
        -> ("squats", "equipment: dumbbells; position: standing")
    "bicep curl with a resistance band on a chair"
        -> ("bicep curl", "equipment: a resistance band; position: a chair")
    "wall push-up" -> ("wall push-up", "")

    Rightmost-match on the separators so exercise names containing these
    words ("step-up onto box") lose as little as possible; anything
    unparsed just stays part of the exercise name, which the models handle
    fine -- this is convenience extraction, not strict grammar.
    """
    remainder = text.strip()

    place = ""
    lower = remainder.lower()
    for sep in (" while ", " on a ", " on an ", " on the ", " on ", " in a ", " in an "):
        idx = lower.rfind(sep)
        if idx != -1:
            place = remainder[idx + len(sep):].strip()
            remainder = remainder[:idx].strip()
            break

    equipment = ""
    lower = remainder.lower()
    idx = lower.rfind(" with ")
    if idx != -1:
        equipment = remainder[idx + len(" with "):].strip()
        remainder = remainder[:idx].strip()

    # Whisper punctuates speech ("Squats, with dumbbells,") -- strip
    # stray commas/periods from each extracted piece.
    exercise = remainder.strip(" ,.;")
    equipment = equipment.strip(" ,.;")
    place = place.strip(" ,.;")

    parts = []
    if equipment:
        parts.append(f"equipment: {equipment}")
    if place:
        parts.append(f"position: {place}")
    return exercise, "; ".join(parts)


def _friendly_error(e: Exception) -> str:
    """Map known failure modes to plain language. A blind or deaf user can't
    as easily get live help troubleshooting a raw Python exception, so the
    primary message needs to be understandable on its own."""
    msg = str(e).lower()
    # Connection checks must come before the "pulled" check: the vision
    # module wraps every failure in a message containing "pulled", so a
    # down server would otherwise be misreported as a missing model.
    if "could not reach ollama" in msg or "ollama serve" in msg or "connect" in msg:
        return "The local AI engine isn't running yet. Ask whoever set up this app to start Ollama."
    if "anthropic_api_key" in msg:
        return "This app isn't fully set up yet -- it's missing an API key."
    if "voice model not found" in msg:
        return "The voice for reading exercises aloud hasn't been downloaded yet."
    if "pulled" in msg and "model" in msg:
        return "The AI model needed for this hasn't been downloaded yet."
    return "Something went wrong on this step. Please try again in a moment."


def _show_error(e: Exception) -> None:
    st.error(_friendly_error(e))
    with st.expander("Technical details"):
        st.code(str(e))


def _select_and_go(name: str) -> None:
    """Quick-select callback: fill the field and generate on the next rerun."""
    st.session_state["audio_exercise"] = name
    st.session_state["auto_trigger"] = True


def _request_voice_retry() -> None:
    """Manual 'try again' (button or T key): speak the retry prompt and
    reopen the microphone on the next rerun."""
    st.session_state["voice_retry_pending"] = True
    st.session_state["voice_retry_count"] = 0
    # Allow an acoustically identical re-recording to be processed.
    st.session_state.pop("last_voice_hash", None)


def _go_stage(stage: str) -> None:
    st.session_state["stage"] = stage
    if stage == "blind_active":
        # One-shot: speak the exercise question and auto-open the mic on
        # the first render of the blind screen only.
        st.session_state["blind_prompt_pending"] = True


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Universal Adaptive Fitness Coach", page_icon=":material/accessibility_new:", layout="centered")

css_path = Path(__file__).parent / "style.css"
st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

col_sr, col_hc = st.columns(2)
with col_sr:
    sr_mode = st.checkbox(
        "Screen reader mode",
        key="sr_mode",
        help="Turn this on if you use a screen reader (NVDA, JAWS, VoiceOver). It silences this "
             "app's own voice so it doesn't talk over your screen reader; announcements go to your "
             "screen reader instead.",
    )
with col_hc:
    high_contrast = st.checkbox("High contrast", key="high_contrast_toggle", help="Strict yellow-on-black WCAG AAA color scheme (7:1+ contrast).")

if high_contrast:
    hc_css_path = Path(__file__).parent / "style_high_contrast.css"
    st.markdown(f"<style>{hc_css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1>
            <svg class="hero-icon" aria-hidden="true" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C13.1 2 14 2.9 14 4C14 5.1 13.1 6 12 6C10.9 6 10 5.1 10 4C10 2.9 10.9 2 12 2Z" fill="#FFFFFF"/>
                <path d="M21 9L15 9.5V22H13V16H11V22H9V9.5L3 9V7L9 7.5C9 6.67 9.67 6 10.5 6H13.5C14.33 6 15 6.67 15 7.5L21 7V9Z" fill="#FFFFFF"/>
            </svg>
            Universal Adaptive Fitness Coach
        </h1>
        <p>Accessible exercise coaching: spoken scripts for blind users, visual form feedback
        for deaf users.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Standing safety notice, shown on every screen. Kept short so it doesn't
# bury the tools; the generated scripts also end with a stop-if-pain step.
st.markdown(
    '<div class="disclaimer">General fitness guidance, <b>not medical advice</b>. '
    "Stop and rest if you feel pain, dizziness, or loss of balance, and check with a "
    "doctor or trainer before starting a new exercise program.</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Feature renderers (shared between modes)
# ---------------------------------------------------------------------------

def _render_voice_status() -> None:
    """A large, centered recording-status panel for the blind screen -- gives
    a sighted helper or low-vision user an unmistakable visual of what phase
    the flow is in, and keeps the voice-first screen from looking like a
    default form. The live recording waveform is the mic widget's own; this
    panel shows the resting/next-step guidance."""
    if "audio_script" in st.session_state:
        dot, title, body = "speaking", "Script ready", (
            "Your script is playing. Press <b>P</b> to pause or resume. "
            "Press <b>Space</b> to record a new exercise."
        )
    else:
        dot, title, body = "ready", "Ready to listen", (
            "Press <b>Space</b>, then say your exercise — for example, "
            "&ldquo;squats, with dumbbells, while standing.&rdquo;"
        )
    st.markdown(
        f'<div class="voice-status" {_live_attrs()}><div class="voice-dot {dot}"></div>'
        f'<div class="voice-status-text"><h3>{title}</h3><p>{body}</p></div></div>',
        unsafe_allow_html=True,
    )


def render_audio_coach(blind: bool = False) -> None:
    sr = _sr_mode()
    with st.container(border=True):
        if sr:
            st.subheader("Audio Coach")
            st.caption("Type an exercise and press Enter (or choose a quick pick). The 10-step "
                       "script appears below for your screen reader to read.")
        elif blind:
            _render_voice_status()
        else:
            st.subheader("Audio Coach")
            st.caption("Generates a 10-step verbal script describing an exercise, with no visual references -- then reads it aloud.")

        # -------------------------------------------------------------------
        # Microphone / voice input -- ONLY when a screen reader is NOT in use.
        # A screen reader captures the Space shortcut (browse mode) so it can't
        # reach our recorder, and screen-reader users type fluently, so voice
        # input is friction, not help, for them. They get the plain form below.
        # -------------------------------------------------------------------
        if not sr:
            if st.session_state.pop("voice_retry_pending", False):
                prompt = NO_AUDIO_TEXT if st.session_state.pop("voice_retry_prompt", "retry") == "no_audio" else RETRY_TEXT
                _speak(prompt)

            voice_clip = st.audio_input(
                "Say your exercise out loud (press to record, press again to stop)",
                key="voice_clip",
            )
            st.button("Didn't catch it? Try again", key="voice_retry_btn", shortcut="T", on_click=_request_voice_retry)
            if voice_clip is not None:
                clip_bytes = voice_clip.getvalue()
                clip_hash = hashlib.sha256(clip_bytes).hexdigest()
                if st.session_state.get("last_voice_hash") != clip_hash:
                    st.session_state["last_voice_hash"] = clip_hash
                    heard = ""
                    heard_level = 0.0
                    with st.spinner("Understanding what you said..."):
                        try:
                            from voice_input import transcribe, audio_level, SILENCE_P99_THRESHOLD
                            heard_level = audio_level(clip_bytes)
                            heard = transcribe(clip_bytes)
                        except Exception as e:
                            _show_error(e)
                    if heard:
                        st.session_state["voice_retry_count"] = 0
                        exercise_heard, context_heard = _parse_voice_request(heard)
                        st.session_state["audio_exercise"] = exercise_heard
                        if context_heard:
                            st.session_state["audio_context"] = context_heard
                        st.session_state["auto_trigger"] = True
                        st.session_state["voice_origin"] = True
                    else:
                        mic_silent = heard_level < SILENCE_P99_THRESHOLD
                        st.session_state["voice_retry_prompt"] = "no_audio" if mic_silent else "retry"
                        retries = st.session_state.get("voice_retry_count", 0)
                        if retries < 2:
                            st.session_state["voice_retry_count"] = retries + 1
                            st.session_state["voice_retry_pending"] = True
                            st.rerun()
                        else:
                            st.warning(
                                "I couldn't make out an exercise across several tries. "
                                + ("Your microphone doesn't seem to be picking up sound -- check it's not "
                                   "muted and the right device is selected. " if mic_silent else "")
                                + "Press T to try again."
                            )

        # -------------------------------------------------------------------
        # Inputs. SR mode: visible and first (type-and-Enter, the natural
        # screen-reader interaction). Self-voicing blind: collapsed (voice is
        # primary). All-tools: inline.
        # -------------------------------------------------------------------
        extras = st.expander("More options — type an exercise, quick picks, or speech speed") if (blind and not sr) else nullcontext()
        with extras:
            st.markdown("**Quick select**" if sr else "**Quick select** (no typing needed)")
            quick_cols = st.columns(len(COMMON_EXERCISES))
            for col, (label, full_name) in zip(quick_cols, COMMON_EXERCISES):
                with col:
                    st.button(label, key=f"quick_{label}", on_click=_select_and_go, args=(full_name,), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                exercise = st.text_input("Exercise name", placeholder="e.g. bodyweight squat", key="audio_exercise")
            with col2:
                context = st.text_input("Context (optional)", placeholder="e.g. beginner, no equipment", key="audio_context")

            use_claude_primary = st.checkbox(
                "Draft directly with Claude (production mode)",
                key="audio_claude_toggle",
                help="Skips the local Ollama draft and generates the approved script in one Claude call, "
                     "using the exact same prompt -- shows this running in production.",
            )

            if not sr:  # pace only matters when the app voices the script
                pace_label = st.select_slider(
                    "Speech pace", options=["Relaxed", "Natural", "Brisk"], value="Relaxed", key="audio_pace",
                    help="Relaxed is slower than natural speech -- easier to follow a physical movement by ear alone.",
                )
            else:
                pace_label = "Relaxed"
        length_scale = {"Relaxed": 1.25, "Natural": 1.0, "Brisk": 0.85}[pace_label]

        generate_clicked = st.button("Generate script" if sr else "Generate & speak", key="audio_generate", icon=":material/auto_awesome:", type="primary")
        auto_triggered = st.session_state.pop("auto_trigger", False)
        voice_origin = st.session_state.pop("voice_origin", False)

        if generate_clicked or auto_triggered:
            approved = None
            if not exercise:
                st.warning("Enter an exercise name first.")
            else:
                with st.spinner("Writing your script..."):
                    try:
                        if use_claude_primary:
                            approved = draft_with_claude(exercise, extra_context=context)
                        else:
                            switch_to(OLLAMA_DRAFT_MODEL)
                            draft = generate_draft(exercise, extra_context=context)
                            approved = review_script(draft, exercise_context=context)
                        st.session_state["audio_script"] = approved
                    except Exception as e:
                        _show_error(e)

            # Synthesize the app's own voice only when NOT using a screen
            # reader; in SR mode the script-card (an ARIA live region) is read
            # by the screen reader, so app audio would just collide with it.
            if approved and not sr:
                with st.spinner("Synthesizing audio..."):
                    try:
                        speak_text = f"Your exercise: {exercise}. {approved}" if voice_origin else approved
                        audio_bytes = synthesize_speech(
                            speak_text, voice_model=DEFAULT_VOICE_MODEL, length_scale=length_scale
                        )
                        time.sleep(0.6)
                        st.session_state["audio_bytes"] = audio_bytes
                        st.session_state["autoplay_pending"] = True
                    except FileNotFoundError as e:
                        st.warning(f"Piper isn't set up yet -- {e}")
                    except RuntimeError as e:
                        _show_error(e)

        if "audio_script" in st.session_state:
            # Flowing read-only text (not a <textarea>, which screen readers
            # read line-by-line in "forms mode"). In SR mode it's a live region
            # so the screen reader reads the whole script when it appears.
            safe_script = html.escape(st.session_state["audio_script"])
            st.markdown(
                f'<div class="script-card" {_live_attrs()}><h4>Your script</h4>{safe_script}</div>',
                unsafe_allow_html=True,
            )

        if "audio_bytes" in st.session_state and not sr:
            play_now = st.session_state.pop("autoplay_pending", False)
            if st.session_state.get("stage") == "blind_active":
                if play_now:
                    _speak(wav=st.session_state["audio_bytes"])
                st.caption("Speaking. Press P to pause or resume the voice.")
            else:
                st.audio(st.session_state["audio_bytes"], format="audio/wav", autoplay=play_now)


def render_form_checker() -> None:
    with st.container(border=True):
        st.subheader("Form Checker")
        st.caption("Snap a photo -- or upload a short video of one rep -- between sets. Get short text corrections. No audio needed.")
        st.caption(
            ":material/info: Your photo or video is analyzed to give form feedback and is **not stored**. "
            "With the local model it never leaves your computer; with the Claude option it is sent to "
            "Anthropic's API for analysis only. Don't upload images of anyone who hasn't agreed to it."
        )

        exercise_v = st.text_input("Exercise being performed", placeholder="e.g. squat", key="vision_exercise")

        # Camera-first for the between-sets flow. The video option exists
        # because form is a movement: one frozen frame can't show descent
        # speed or where in the rep the form breaks down.
        photo_source = st.radio(
            "Source",
            ["Take a photo", "Upload a photo", "Upload a video (one rep)"],
            horizontal=True,
            key="vision_source",
        )
        if photo_source == "Take a photo":
            uploaded = st.camera_input("Camera -- take a photo of your form", key="vision_camera")
        elif photo_source == "Upload a photo":
            uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="vision_upload")
        else:
            uploaded = st.file_uploader(
                "Upload a short video of one repetition (a few seconds)",
                type=["mp4", "mov", "webm"],
                key="vision_video",
            )

        use_claude_vision = st.checkbox(
            "Analyze with Claude instead of local model",
            key="vision_claude_toggle",
            help="Sends the same prompt to Claude's vision API instead of the local Ollama vision model.",
        )

        # Tested finding, not a guess: on frame sequences the local model
        # fabricates confident corrections even when no person is visible --
        # Claude abstains correctly on the same frames.
        if photo_source == "Upload a video (one rep)" and not use_claude_vision:
            st.info(
                "For video, the local model is unreliable at admitting when it can't see the "
                "movement clearly -- it may give confident but wrong advice. The Claude option "
                "above is strongly recommended for video analysis."
            )

        if uploaded and photo_source == "Upload a photo":
            st.image(uploaded, caption="Uploaded photo", width=300)

        if st.button("Check my form", key="vision_analyze", disabled=uploaded is None, icon=":material/search:", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    exercise_name = exercise_v or "exercise"
                    if photo_source == "Upload a video (one rep)":
                        frames = extract_frames(uploaded.getvalue())
                        if use_claude_vision:
                            captions = check_form_sequence_with_claude(frames, exercise_name)
                        else:
                            captions = check_form_sequence(frames, exercise_name)
                    else:
                        image_bytes = uploaded.getvalue()
                        if use_claude_vision:
                            captions = check_form_with_claude(image_bytes, exercise_name)
                        else:
                            captions = check_form(image_bytes, exercise_name)
                    st.session_state["vision_captions"] = captions
                except ValueError as e:
                    # extract_frames raises these with user-facing wording.
                    st.error(str(e))
                except Exception as e:
                    _show_error(e)

        if "vision_captions" in st.session_state:
            safe_captions = html.escape(st.session_state["vision_captions"])
            st.markdown(
                f'<div class="result-card"><h4>Corrections</h4>{safe_captions}</div>',
                unsafe_allow_html=True,
            )


def render_live_tracker() -> None:
    with st.container(border=True):
        st.subheader("Live Tracker")
        st.caption(
            "Real-time squat coaching for squats: skeleton overlay, depth, and a rep counter "
            "that also checks each rep for depth, knee tracking, and tempo. Feedback shows as "
            "big text AND is spoken aloud, so it works for blind and deaf users alike. Control "
            "your set hands-free -- cross your arms to end a set, wave both hands overhead to "
            "start the next. Runs entirely in your browser: no frames ever leave your machine."
        )
        st_html(POSE_TRACKER_HTML, height=POSE_TRACKER_HEIGHT)


def render_start_over() -> None:
    # Screen readers capture the Escape shortcut, so drop it in SR mode and
    # rely on standard Tab-then-Enter activation.
    st.button("Change mode (blind or deaf)", key="start_over",
              shortcut=(None if _sr_mode() else "Escape"), on_click=_go_stage, args=("q1",))


# ---------------------------------------------------------------------------
# Onboarding stage machine
# ---------------------------------------------------------------------------

stage = st.session_state.setdefault("stage", "q1")

if stage == "q1":
    with st.container(border=True):
        st.subheader("Welcome")
        if _sr_mode():
            instr = "Choose the button that fits you, then press Enter."
        else:
            instr = "Press the SPACE BAR if you are blind.\nPress the letter D if you are deaf."
        st.markdown(
            f'<div class="script-card" {_live_attrs()}><h4>Are you blind or deaf?</h4>{instr}</div>',
            unsafe_allow_html=True,
        )
        # No-op in screen-reader mode (the card above is read by the screen
        # reader). May be silent on a fresh load until the first gesture.
        _speak(Q1_TEXT)

        # In screen-reader mode the Space/D shortcuts don't work (the screen
        # reader captures those keys), and their badges just mislead -- so
        # drop them and rely on the standard Tab-then-Enter activation.
        col_b, col_d = st.columns(2)
        with col_b:
            st.button("I am blind", key="q1_blind", shortcut=(None if _sr_mode() else "Space"),
                      type="primary", on_click=_go_stage, args=("blind_active",), use_container_width=True)
        with col_d:
            st.button("I am deaf", key="q1_deaf", shortcut=(None if _sr_mode() else "D"),
                      on_click=_go_stage, args=("deaf_active",), use_container_width=True)

        st.button("Skip — show all tools", key="q1_skip", type="tertiary",
                  on_click=_go_stage, args=("all_active",))

elif stage == "blind_active":
    if _sr_mode():
        # Screen-reader users get a plain, standard, labeled form: Tab to the
        # exercise field, type, press Enter, and the screen reader reads the
        # script. NO self-voicing, NO mic, NO custom key shortcuts -- that
        # model is for users WITHOUT a screen reader.
        render_start_over()
        render_audio_coach(blind=False)
    else:
        # Self-voicing, voice-first flow (no screen reader): spoken prompt,
        # gesture-driven mic (Space), custom shortcuts.
        if st.session_state.pop("blind_prompt_pending", False):
            _speak(Q3_TEXT)
        _blind_keys_component()
        render_start_over()
        render_audio_coach(blind=True)

elif stage == "deaf_active":
    tab_vision, tab_live = st.tabs([
        ":material/photo_camera: Form Checker",
        ":material/directions_run: Live Tracker",
    ])
    with tab_vision:
        render_form_checker()
    with tab_live:
        render_live_tracker()
    render_start_over()

else:  # all_active -- everything, like the pre-onboarding app
    tab_audio, tab_vision, tab_live = st.tabs([
        ":material/record_voice_over: Audio Coach (Blind)",
        ":material/photo_camera: Form Checker (Deaf)",
        ":material/directions_run: Live Tracker (Deaf)",
    ])
    with tab_audio:
        render_audio_coach()
    with tab_vision:
        render_form_checker()
    with tab_live:
        render_live_tracker()
    render_start_over()
