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


def preprocess(samples: np.ndarray, trim_db: float = 40.0) -> np.ndarray:
    """Trim leading/trailing silence and normalize safely."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return samples

    trimmed, _ = librosa.effects.trim(samples, top_db=trim_db)
    peak = float(np.max(np.abs(trimmed))) if trimmed.size else 0.0
    if peak > 0:
        trimmed = trimmed / peak
    return trimmed.astype(np.float32, copy=False)
