"""
video_frames.py

Extracts a handful of evenly spaced frames from a short uploaded video so
the Form Checker can assess a *movement* rather than a single frozen
moment -- a rep's problems often show up in the middle of the descent or at
the bottom position, which one photo can't capture.

This is deliberately still not live tracking: the user records one rep on
their phone (a few seconds), uploads it, and the sampled frames go to the
vision model as an ordered sequence in a single request. Vision LLM latency
(seconds per request) makes this the honest middle ground between "one
photo" and real-time analysis, which would need pose estimation instead.
"""

import tempfile
from pathlib import Path

import cv2

# 6 frames spans a single rep well (start, descent x2, bottom, ascent, end)
# without bloating the vision request. Both qwen2.5vl and Claude handle a
# multi-image message of this size comfortably.
DEFAULT_FRAME_COUNT = 6

# Longest clip we'll accept, to keep "one rep" honest and requests bounded.
MAX_VIDEO_SECONDS = 30


def extract_frames(video_bytes: bytes, frame_count: int = DEFAULT_FRAME_COUNT) -> list[bytes]:
    """
    Pull evenly spaced JPEG frames from a video.

    Args:
        video_bytes: raw bytes of the uploaded video file (mp4/mov/webm).
        frame_count: how many frames to sample across the clip's duration.

    Returns:
        List of JPEG-encoded frame bytes, in chronological order.

    Raises:
        ValueError: unreadable video, or longer than MAX_VIDEO_SECONDS.
    """
    # OpenCV can't read from memory -- it needs a real file path, so write a
    # temp file for the duration of the read. delete=False because Windows
    # won't let cv2 open a file that's still held open by this process.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = Path(tmp.name)

    try:
        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            raise ValueError("Couldn't read that video. Try an .mp4 recorded on your phone.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if total_frames <= 0:
            raise ValueError("That video appears to be empty.")

        duration = total_frames / fps
        if duration > MAX_VIDEO_SECONDS:
            raise ValueError(
                f"That video is {duration:.0f} seconds long -- please trim it to a single "
                f"repetition (under {MAX_VIDEO_SECONDS} seconds)."
            )

        # Evenly spaced indices, inset slightly from the very first/last
        # frame (those are often mid-fumble with the phone).
        n = min(frame_count, total_frames)
        indices = [int(total_frames * (i + 0.5) / n) for i in range(n)]

        frames: list[bytes] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            # Cap the long edge at 1024px -- plenty for form assessment,
            # keeps the multi-image request small.
            h, w = frame.shape[:2]
            scale = 1024 / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                frames.append(jpg.tobytes())

        cap.release()

        if not frames:
            raise ValueError("Couldn't extract any frames from that video.")
        return frames
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import numpy as np

    # Self-test: build a tiny synthetic video, then sample it.
    test_path = Path("_test_video.mp4")
    writer = cv2.VideoWriter(str(test_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for i in range(90):  # 3 seconds
        frame = np.full((240, 320, 3), i * 2 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    frames = extract_frames(test_path.read_bytes())
    print(f"extracted {len(frames)} frames, sizes: {[len(f) for f in frames]}")
    test_path.unlink()
