from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass(frozen=True)
class PitchTrack:
    times: np.ndarray
    frequencies_hz: np.ndarray
    voiced: np.ndarray
    confidence: np.ndarray


def detect_pitch(
    samples: np.ndarray,
    sample_rate: int,
    *,
    fmin: float = 65.41,
    fmax: float = 1046.50,
    frame_length: int = 2048,
    hop_length: int = 256,
) -> PitchTrack:
    """Estimate monophonic fundamental frequency with librosa.pyin."""
    if len(samples) < frame_length:
        raise ValueError("Audio is too short for pitch analysis")

    f0, voiced_flag, voiced_prob = librosa.pyin(
        samples,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sample_rate, hop_length=hop_length)
    frequencies = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    voiced = np.asarray(voiced_flag, dtype=bool)
    confidence = np.nan_to_num(voiced_prob, nan=0.0).astype(np.float32)
    return PitchTrack(times, frequencies, voiced, confidence)
