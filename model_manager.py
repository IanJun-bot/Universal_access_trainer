"""
model_manager.py

Keeps Ollama's VRAM usage sequential and predictable: before switching to a
new model, unload whatever else is currently resident. This avoids gambling
on the text-draft model and the vision model fitting in VRAM at the same
time — on an 8GB card that math is tight, not guaranteed.

This uses Ollama's `keep_alive=0` mechanism, which unloads a model from
memory immediately after its next response. That is the only reliable way
to free VRAM held by the Ollama server process.

Important: clearing memory in your own Python process (e.g.
`torch.cuda.empty_cache()`) has NO effect on Ollama's VRAM. Ollama runs as
its own background server with its own GPU memory space — your script
talks to it over HTTP, it doesn't share memory with it.
"""

import ollama


def get_loaded_models() -> list[str]:
    """Return the names of models currently resident in Ollama's memory."""
    try:
        resp = ollama.ps()
        models = resp.get("models", []) if isinstance(resp, dict) else getattr(resp, "models", [])
        names = []
        for m in models:
            name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
            if name:
                names.append(name)
        return names
    except Exception:
        return []


def unload_model(model_name: str) -> None:
    """Force-unload a specific model that is currently loaded."""
    try:
        ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": ""}],
            keep_alive=0,
        )
    except Exception:
        # If it wasn't actually loaded, this may no-op or error — safe to ignore either way.
        pass


def switch_to(target_model: str) -> None:
    """
    Ensure only target_model occupies GPU memory going forward.

    Unloads every other currently-loaded model first, so the audio-draft
    model and the vision model are never resident at the same time.
    Call this before any ollama.chat() call where VRAM headroom matters.
    """
    for loaded in get_loaded_models():
        if loaded != target_model:
            unload_model(loaded)


if __name__ == "__main__":
    print("Currently loaded:", get_loaded_models())
