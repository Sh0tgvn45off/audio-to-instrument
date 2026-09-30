from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int


def load_audio(path: str, target_sr: int | None = 22050) -> AudioData:
    """Load a mono audio file and optionally resample it."""
    samples, sample_rate = librosa.load(path, sr=target_sr, mono=True)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise ValueError("Audio file contains no samples")
    return AudioData(samples=samples, sample_rate=sample_rate)


def preprocess(
    samples: np.ndarray,
    *,
    trim_db: float | None = None,
) -> np.ndarray:
    """Normalize audio without changing its timeline.

    By default the sample count is preserved. This is important for
    transcription because removing leading silence would otherwise shift
    every detected MIDI timestamp relative to the original recording.

    Optional trimming is retained for offline experiments, but callers that
    need MIDI aligned to the source recording should leave trim_db as None.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return samples

    processed = samples
    if trim_db is not None:
        processed, _ = librosa.effects.trim(processed, top_db=trim_db)

    peak = float(np.max(np.abs(processed))) if processed.size else 0.0
    if peak > 0:
        processed = processed / peak
    return processed.astype(np.float32, copy=False)
