"""
voice_input.py

Local speech-to-text for the Audio Coach's voice-first flow: a blind user
records a short clip via the microphone widget ("bodyweight squat,
beginner"), this transcribes it, and the app fills the exercise field and
generates immediately -- speak, then hear. No typing, no visual scanning.

Uses faster-whisper (CTranslate2 Whisper) on CPU so it never competes with
Ollama for GPU memory, mirroring the same reasoning as the Piper TTS choice.
The model (~74MB for "base") downloads automatically on first use and is
cached by huggingface_hub after that.
"""

import io
import wave

import numpy as np
from faster_whisper import WhisperModel

# A recording whose 99th-percentile absolute level (0..1 scale) is below
# this is treated as silence without even asking Whisper -- which matters
# because Whisper famously HALLUCINATES on silence, confidently transcribing
# "Thank you." out of nothing.
#
# Why the 99th percentile and NOT the mean: a short word ("squat") inside a
# few seconds of recording has a high PEAK but a low MEAN, because the
# surrounding silence drags the average down. An earlier mean-based gate
# therefore rejected real speech at normal-but-quiet mic levels (a laptop
# mic peaking ~0.08 full-scale) -- the "mic isn't picking up my voice" bug.
# The 99th percentile tracks sustained speech energy, ignores lone
# clicks/pops, and still clears room noise. This gate only rejects
# effectively-dead audio; the VAD filter and blocklist below handle the rest.
SILENCE_P99_THRESHOLD = 0.008

# Classic Whisper silence-hallucinations, matched case-insensitively against
# the whole (stripped) transcript.
HALLUCINATION_BLOCKLIST = {
    "thank you", "thanks", "thank you for watching", "thanks for watching",
    "you", "bye", "bye-bye", "goodbye", "the end", "so", "okay", "oh",
}

# "base" is the sweet spot for short command-style phrases: noticeably more
# accurate than "tiny" on exercise vocabulary, still transcribes a few
# seconds of audio in well under a second on CPU int8.
WHISPER_MODEL_SIZE = "base"

_model_cache: dict = {}


def _get_model() -> WhisperModel:
    """Load (and cache) the Whisper model so repeated calls don't reload it."""
    if WHISPER_MODEL_SIZE not in _model_cache:
        _model_cache[WHISPER_MODEL_SIZE] = WhisperModel(
            WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"
        )
    return _model_cache[WHISPER_MODEL_SIZE]


def transcribe(audio_bytes: bytes) -> str:
    """
    Transcribe recorded speech to text.

    Args:
        audio_bytes: raw audio file bytes (WAV, as produced by
            st.audio_input's UploadedFile.getvalue()).

    Returns:
        The transcribed text, stripped of surrounding whitespace and any
        trailing sentence punctuation (Whisper adds a period to short
        commands, which would otherwise end up in the exercise field).
        Empty string if nothing intelligible was said -- including recordings
        that are silent or contain only a known silence-hallucination.
    """
    # Energy gate first: a silent room should never reach the model at all.
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if pcm.size == 0:
            return ""
        if float(np.percentile(np.abs(pcm), 99)) / 32768.0 < SILENCE_P99_THRESHOLD:
            return ""
    except Exception:
        pass  # not a readable WAV -- let Whisper try anyway

    model = _get_model()
    segments, _info = model.transcribe(
        io.BytesIO(audio_bytes),
        language="en",
        beam_size=5,
        # Drop non-speech audio before decoding -- the main defense against
        # Whisper inventing text from breath noise and room tone.
        vad_filter=True,
        # Biases decoding toward exercise vocabulary -- without this, short
        # clips mishear domain words ("squat" -> "squad").
        initial_prompt=(
            "Exercise names: squat, bodyweight squat, lunge, push-up, plank, "
            "bicep curl, shoulder press, calf raise, glute bridge, stretch, "
            "step-up, lateral raise, wood chop, arm circles."
        ),
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    text = text.rstrip(".!?,;: ").strip()
    if text.lower() in HALLUCINATION_BLOCKLIST:
        return ""
    return text


def audio_level(audio_bytes: bytes) -> float:
    """
    Return a WAV clip's 99th-percentile absolute amplitude on a 0..1 scale.

    Used to tell two failure modes apart when transcription comes back empty:
    a near-zero level means the microphone captured essentially no sound (a
    hardware/selection/permission problem worth telling the user about),
    whereas a healthy level means we heard something but couldn't parse an
    exercise from it (a "say it again, slower" situation). A blind user
    can't see a level meter, so this distinction has to be spoken.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if pcm.size == 0:
            return 0.0
        return float(np.percentile(np.abs(pcm), 99)) / 32768.0
    except Exception:
        return 0.0


if __name__ == "__main__":
    # Self-test that needs no microphone: synthesize a phrase with Piper,
    # then check Whisper can hear it back.
    from tts_piper import synthesize_speech

    wav = synthesize_speech("bodyweight squat")
    heard = transcribe(wav)
    print(f"Piper said 'bodyweight squat' -> Whisper heard: {heard!r}")
