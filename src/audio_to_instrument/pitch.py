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
    confidence_threshold: float = 0.65,
) -> PitchTrack:
    """Estimate monophonic fundamental frequency with librosa.pyin.

    Frames below confidence_threshold are treated as unvoiced to suppress
    unstable estimates from consonants, breath noise, and weak frames.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if len(samples) < frame_length:
        raise ValueError("Audio is too short for pitch analysis")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")

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
    confidence = np.nan_to_num(voiced_prob, nan=0.0).astype(np.float32)

    voiced = np.asarray(voiced_flag, dtype=bool) & (confidence >= confidence_threshold)
    frequencies[~voiced] = 0.0

    return PitchTrack(
        times=np.asarray(times, dtype=np.float32),
        frequencies_hz=frequencies,
        voiced=voiced,
        confidence=confidence,
    )
