# tts/speaker.py
# Nova's mouth. Speaks a reply *while it is still being written*: each sentence
# is synthesized as soon as the brain finishes it and played the moment the one
# before ends, so Nova starts talking after the first sentence instead of after
# the whole reply. With barge-in on (Piper only), talking over Nova stops the
# audio AND tells the brain to stop generating the rest.

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Callable, Iterable, Optional

import numpy as np

from config import (
    SAMPLE_RATE,
    MIC_INDEX,
    TTS_BACKEND,
    USE_BARGE_IN,
    BARGE_RMS_THRESHOLD,
)

FRAME = 1280               # 80 ms @ 16 kHz per mic read
_BARGE_MIN_FRAMES = 4      # ~320 ms of sustained speech before we count it as barge-in
_BARGE_GRACE_SEC = 0.35    # ignore the very start of the reply (playback ramp-up / mic settling)
_BARGE_DEBUG = os.getenv("BARGE_DEBUG") == "1"  # print peak mic level to help tune the threshold
_barge_warned = False      # only warn once if the monitor mic can't open
_DONE = object()           # end-of-reply marker on the clip queue


def _rms(frame_i16: np.ndarray) -> float:
    f = frame_i16.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(f * f))) if f.size else 0.0


def speak(text: str, barge_in: Optional[bool] = None) -> bool:
    """Speak one piece of text. Returns True if the user interrupted."""
    return speak_stream([text], barge_in=barge_in)


def speak_stream(
    sentences: Iterable[str],
    on_sentence: Optional[Callable[[str], None]] = None,
    on_interrupt: Optional[Callable[[], None]] = None,
    barge_in: Optional[bool] = None,
) -> bool:
    """Speak sentences as they arrive (e.g. straight from the brain's stream).
    on_sentence(text) fires as each one starts playing (for printing in step
    with the voice); on_interrupt() fires on barge-in (to stop generation).
    Returns True if the user talked over Nova."""
    if TTS_BACKEND != "piper":
        return _speak_plain(sentences, on_sentence)
    if barge_in is None:
        barge_in = USE_BARGE_IN
    return _speak_piper(sentences, on_sentence, on_interrupt, barge_in)


def _speak_plain(sentences, on_sentence) -> bool:
    """ElevenLabs or text-only: say each sentence as it arrives (no barge-in)."""
    from tts.voice import speak as say

    for text in sentences:
        if on_sentence:
            on_sentence(text)
        if TTS_BACKEND != "none":  # "none" = the printed line is the reply
            say(text)
    return False


def _open_monitor_mic(sd):
    global _barge_warned
    try:
        mic = sd.InputStream(
            samplerate=SAMPLE_RATE, device=MIC_INDEX, channels=1,
            dtype="int16", blocksize=FRAME,
        )
        mic.start()
        return mic
    except Exception as e:
        if not _barge_warned:
            print(f"   [barge-in disabled: couldn't open the monitor mic ({e}). "
                  f"Set USE_BARGE_IN=0 to silence this, or use headphones.]")
            _barge_warned = True
        return None


def _still_playing(sd, clip_end: float) -> bool:
    """True while the current clip is audible. Asks PortAudio, because the wall
    clock alone would clip each sentence's tail by the output latency; the wall
    clock is only a backstop in case the stream never reports done."""
    if time.time() > clip_end + 1.0:
        return False
    try:
        return bool(sd.get_stream().active)
    except Exception:
        return time.time() < clip_end


def _speak_piper(sentences, on_sentence, on_interrupt, barge_in: bool) -> bool:
    import sounddevice as sd
    from tts.piper_tts import synthesize

    clips: queue.Queue = queue.Queue()
    stop = threading.Event()

    def produce() -> None:
        # Background: pull sentences (this is where the LLM streams) and
        # synthesize each one, so its audio is ready the moment it's needed.
        it = iter(sentences)
        try:
            for text in it:
                if stop.is_set():
                    break
                try:
                    samples, rate = synthesize(text)
                except Exception as e:
                    print(f"[Piper TTS error] {e}")
                    samples, rate = None, 0  # still print the text
                clips.put((text, samples, rate))
        except Exception as e:
            print(f"[speaker] reply stream failed: {e}")
        finally:
            close = getattr(it, "close", None)
            if close:
                try:
                    close()  # lets the brain save the (possibly partial) reply
                except Exception:
                    pass
            clips.put(_DONE)

    threading.Thread(target=produce, name="nova-tts", daemon=True).start()

    mic = _open_monitor_mic(sd) if barge_in else None
    finished = interrupted = False
    speech_start = 0.0  # when Nova's first clip started
    clip_end = 0.0      # when the playing clip ends (0 = silent)
    loud, peak = 0, 0.0
    try:
        while True:
            if clip_end and not _still_playing(sd, clip_end):
                clip_end = 0.0

            if not clip_end:  # silent: start the next clip if one is ready
                if finished:
                    break
                try:
                    item = clips.get_nowait()
                except queue.Empty:
                    item = None
                if item is _DONE:
                    finished = True
                    continue
                if item is not None:
                    text, samples, rate = item
                    if on_sentence:
                        on_sentence(text)
                    if samples is not None and samples.size:
                        sd.play(samples, samplerate=rate)
                        speech_start = speech_start or time.time()
                        clip_end = time.time() + samples.size / float(rate)
                    continue

            # Pace the loop. With a monitor mic each ~80 ms read doubles as the
            # barge-in sample. Only speech over Nova's *playing* voice counts, and
            # a sustained run of it carries across sentence boundaries.
            if mic is None:
                time.sleep(0.02)
                continue
            data, _ = mic.read(FRAME)
            if not clip_end or time.time() - speech_start < _BARGE_GRACE_SEC:
                continue
            level = _rms(data[:, 0])
            peak = max(peak, level)
            if level >= BARGE_RMS_THRESHOLD:
                loud += 1
                if loud >= _BARGE_MIN_FRAMES:
                    interrupted = True
                    break
            else:
                loud = 0
    except Exception as e:
        print(f"[speaker] playback error: {e}")
    finally:
        if not finished:  # barge-in or error: silence Nova and stop the reply
            stop.set()
            sd.stop()
            if on_interrupt:
                on_interrupt()
        if mic is not None:
            mic.stop()
            mic.close()

    if _BARGE_DEBUG and mic is not None:
        if interrupted:
            print(f"   [barge-in fired: peak {peak:.3f} >= {BARGE_RMS_THRESHOLD}]")
        else:
            print(f"   [barge-in: peak mic level {peak:.3f} (threshold {BARGE_RMS_THRESHOLD}) "
                  f"— lower BARGE_RMS_THRESHOLD if you spoke and it didn't stop]")
    return interrupted
