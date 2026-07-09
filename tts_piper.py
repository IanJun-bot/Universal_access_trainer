"""
tts_piper.py

Converts the final, approved script text into audio using Piper (a fast,
local, CPU-based TTS engine -- no VRAM needed, so it runs alongside Ollama
without competing for GPU memory).

Synthesizes straight to an in-memory WAV buffer (no temp file on disk).
Two audio-quality fixes live here, both discovered by ear and confirmed by
measurement:
  1. Sentence gaps: Piper's high-level synthesize_wav() writes sentence
     chunks back-to-back with zero gap (an audible stutter between numbered
     steps) -- we insert SENTENCE_SILENCE_SECONDS of silence, with a 10ms
     fade at each chunk edge so the splice doesn't click.
  2. Loudness: Piper's default normalizes EACH sentence chunk to full scale
     independently, amplifying short fragments and trailing breaths up to
     ~8x more than neighboring sentences -- heard as a garbled "moan" and
     per-sentence volume whiplash. We synthesize at natural levels and
     normalize the whole utterance once, to 0.85 peak.

Setup (Windows):
    1. Install Piper directly with pip (already in requirements.txt):
           pip install piper-tts
    2. Download a voice model with Piper's built-in downloader. This saves
       the .onnx and .onnx.json files into your current directory:
           python -m piper.download_voices en_GB-cori-medium
    That's it -- no separate .exe, no manual path configuration.
"""

import io
import wave

import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig

DEFAULT_VOICE_MODEL = "en_GB-cori-medium.onnx"
SENTENCE_SILENCE_SECONDS = 0.4

# Piper's length_scale stretches phoneme duration: 1.0 is the voice's natural
# pace, higher is slower. Exercise instructions need to be followed by ear in
# real time (find the position, hold it, move on) -- natural conversational
# pace is too fast for that, so 1.25 is the default rather than 1.0.
DEFAULT_LENGTH_SCALE = 1.25

_voice_cache = {}


def _get_voice(voice_model: str) -> PiperVoice:
    """Load (and cache) a Piper voice model so repeated calls don't reload it from disk."""
    if voice_model not in _voice_cache:
        from pathlib import Path

        model_path = Path(voice_model)
        if not model_path.is_absolute():
            model_path = Path(__file__).parent / model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"Voice model not found at {voice_model}. Download it with:\n"
                f"    python -m piper.download_voices en_GB-cori-medium"
            )
        _voice_cache[voice_model] = PiperVoice.load(str(model_path))
    return _voice_cache[voice_model]


def synthesize_speech(
    text: str,
    voice_model: str = DEFAULT_VOICE_MODEL,
    length_scale: float = DEFAULT_LENGTH_SCALE,
) -> bytes:
    """
    Convert text to WAV audio using Piper, entirely in memory.

    Args:
        text: the approved script to speak
        voice_model: path to the .onnx voice model (its matching .onnx.json
            must sit alongside it)
        length_scale: phoneme duration multiplier (1.0 = natural pace,
            >1.0 = slower). See DEFAULT_LENGTH_SCALE for why this defaults
            to slower-than-natural.

    Returns:
        Raw WAV file bytes, ready to hand to st.audio() or write to disk.
    """
    voice = _get_voice(voice_model)
    # normalize_audio=False is the fix for an audible glitch: Piper's default
    # normalizes EACH sentence chunk to full scale independently, so a short
    # fragment or trailing breath gets amplified up to ~8x more than the
    # sentence before it -- heard as a loud, garbled "moan" and per-sentence
    # loudness whiplash. Keep natural relative levels and normalize the whole
    # utterance once at the end instead.
    syn_config = SynthesisConfig(length_scale=length_scale, normalize_audio=False)

    sample_rate = None
    pieces: list[np.ndarray] = []
    for audio_chunk in voice.synthesize(text, syn_config):
        sample_rate = audio_chunk.sample_rate
        arr = np.asarray(audio_chunk.audio_float_array, dtype=np.float32).copy()
        # 10ms fade at each chunk edge: the silence gaps are spliced in as
        # raw zeros, and a chunk ending off-zero produces an audible click
        # at every sentence boundary without this.
        fade = int(sample_rate * 0.010)
        if arr.size > 2 * fade > 0:
            arr[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            arr[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        if pieces:
            pieces.append(np.zeros(int(sample_rate * SENTENCE_SILENCE_SECONDS), dtype=np.float32))
        pieces.append(arr)

    if not pieces or sample_rate is None:
        raise RuntimeError("Piper produced no audio for this text.")

    full = np.concatenate(pieces)
    # One global normalization: consistent volume without distorting the
    # natural loudness relationships between sentences.
    peak = float(np.abs(full).max())
    if peak > 1e-6:
        full = full * (0.85 / peak)
    pcm = (np.clip(full, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setframerate(sample_rate)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        wav_file.writeframes(pcm)

    return buffer.getvalue()


if __name__ == "__main__":
    audio_bytes = synthesize_speech(
        "Step one. Stand with your feet hip width apart, at twelve o'clock."
    )
    with open("test_output.wav", "wb") as f:
        f.write(audio_bytes)
    print("Wrote test_output.wav")
