from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass(frozen=True)
class ArticulationFeatures:
    """Frame-level acoustic evidence used for note boundary decisions."""

    times: np.ndarray
    rms_db: np.ndarray
    onset_strength: np.ndarray


def detect_articulation_features(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_length: int = 2048,
    hop_length: int = 256,
) -> ArticulationFeatures:
    """Extract continuous acoustic features for articulation-aware segmentation.

    RMS is used as release evidence, while spectral-flux onset strength is
    retained as supporting attack evidence. Neither signal is a hard note
    boundary by itself.
    """
    if len(samples) == 0:
        empty = np.array([], dtype=np.float32)
        return ArticulationFeatures(empty, empty, empty)

    rms = librosa.feature.rms(
        y=samples,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0].astype(np.float32)

    onset = librosa.onset.onset_strength(
        y=samples,
        sr=sample_rate,
        hop_length=hop_length,
        center=True,
    ).astype(np.float32)

    times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sample_rate,
        hop_length=hop_length,
    ).astype(np.float32)

    peak = float(np.max(rms)) if len(rms) else 0.0
    if peak <= 0.0:
        rms_db = np.full(len(rms), -80.0, dtype=np.float32)
    else:
        rms_db = (
            20.0 * np.log10(np.maximum(rms, np.finfo(np.float32).tiny) / peak)
        ).astype(np.float32)

    onset = onset - float(np.min(onset)) if len(onset) else onset
    onset_peak = float(np.max(onset)) if len(onset) else 0.0
    if onset_peak > 0.0:
        onset = onset / onset_peak
    else:
        onset = np.zeros_like(onset)

    return ArticulationFeatures(times, rms_db, onset.astype(np.float32))


def sample_articulation_features(
    time: float,
    features: ArticulationFeatures,
) -> tuple[float, float]:
    """Return RMS dB and onset strength at the frame nearest to time."""
    if len(features.times) == 0:
        return 0.0, 0.0

    index = int(np.searchsorted(features.times, time))
    if index <= 0:
        index = 0
    elif index >= len(features.times):
        index = len(features.times) - 1
    elif abs(float(features.times[index]) - time) >= abs(
        float(features.times[index - 1]) - time
    ):
        index -= 1

    return float(features.rms_db[index]), float(features.onset_strength[index])
