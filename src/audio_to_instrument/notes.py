from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .pitch import PitchTrack


@dataclass(frozen=True)
class Note:
    midi_note: int
    start: float
    end: float
    velocity: int = 100

    @property
    def duration(self) -> float:
        return self.end - self.start


def hz_to_midi(frequency_hz: float) -> float:
    """Convert frequency in Hz to a fractional MIDI note number."""
    if frequency_hz <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(frequency_hz / 440.0)


def detect_onsets(
    samples: np.ndarray,
    sample_rate: int,
    *,
    hop_length: int = 256,
    backtrack: bool = True,
) -> np.ndarray:
    """Detect likely note attacks in a monophonic recording."""
    if len(samples) == 0:
        return np.array([], dtype=np.float32)

    onset_frames = librosa.onset.onset_detect(
        y=samples,
        sr=sample_rate,
        hop_length=hop_length,
        backtrack=backtrack,
        units="frames",
    )
    return librosa.frames_to_time(
        onset_frames, sr=sample_rate, hop_length=hop_length
    ).astype(np.float32)


def detect_energy_regions(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_length: int = 2048,
    hop_length: int = 256,
    top_db: float = 35.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frame times and an energy-based voiced/silent mask.

    The mask is deliberately separate from pitch tracking. It lets note
    segmentation preserve short gaps and releases even when pitch remains
    unchanged.
    """
    if len(samples) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=bool)

    rms = librosa.feature.rms(
        y=samples,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sample_rate, hop_length=hop_length
    )

    peak = float(np.max(rms)) if len(rms) else 0.0
    if peak <= 0.0:
        return times.astype(np.float32), np.zeros(len(rms), dtype=bool)

    threshold = peak * (10.0 ** (-top_db / 20.0))
    return times.astype(np.float32), (rms >= threshold)


def track_to_notes(
    track: PitchTrack,
    *,
    onset_times: np.ndarray | None = None,
    energy_times: np.ndarray | None = None,
    energy_voiced: np.ndarray | None = None,
    min_duration: float = 0.05,
    max_gap: float = 0.06,
    cents_tolerance: float = 80.0,
    change_frames: int = 3,
) -> list[Note]:
    """Convert a monophonic pitch track into articulation-aware MIDI notes.

    Pitch determines note identity. Pitch changes, detected attacks, and
    energy gaps determine boundaries. This preserves repeated notes at the
    same pitch and short staccato gaps instead of merging them automatically.
    """
    if change_frames < 1:
        raise ValueError("change_frames must be at least 1")
    if min_duration < 0:
        raise ValueError("min_duration cannot be negative")

    if len(track.times) == 0:
        return []

    raw_midi = np.array(
        [
            hz_to_midi(float(f)) if v else np.nan
            for f, v in zip(track.frequencies_hz, track.voiced)
        ],
        dtype=np.float32,
    )
    smoothed = _median_smooth(raw_midi, window=5)

    # Kept in the API for compatibility and future attack-time refinement.
    # Current segmentation deliberately does not force boundaries at every
    # onset because onset detectors can fire inside sustained notes.
    onset_times = (
        np.asarray(onset_times, dtype=np.float32)
        if onset_times is not None
        else np.array([], dtype=np.float32)
    )

    energy_times = (
        np.asarray(energy_times, dtype=np.float32)
        if energy_times is not None
        else np.array([], dtype=np.float32)
    )
    energy_voiced = (
        np.asarray(energy_voiced, dtype=bool)
        if energy_voiced is not None
        else np.array([], dtype=bool)
    )

    notes: list[Note] = []
    active_start: float | None = None
    active_midis: list[float] = []
    pending_midis: list[float] = []
    pending_start: float | None = None
    last_voiced_time: float | None = None
    previous_time: float | None = None
    for time, value, is_voiced in zip(track.times, smoothed, track.voiced):
        time = float(time)

        # Onsets are evidence of an attack, not automatic note boundaries.
        # A sustained note can contain transient peaks, consonants, or vibrato
        # that trigger onset detection. Only an actual energy/pitch gap closes
        # the current note, preventing sustained notes from being fragmented.
        energy_active = _energy_is_active(
            time, energy_times, energy_voiced
        )

        if (
            not is_voiced
            or not np.isfinite(value)
            or (len(energy_times) and not energy_active)
        ):
            if active_start is not None and last_voiced_time is not None:
                gap = time - last_voiced_time
                if gap > max_gap or (
                    len(energy_times) and not energy_active and gap > 0.02
                ):
                    _finish_note(
                        notes,
                        active_start,
                        last_voiced_time,
                        active_midis,
                        min_duration,
                    )
                    active_start = None
                    active_midis = []
                    pending_midis = []
                    pending_start = None
            previous_time = time
            continue

        midi = float(value)

        if active_start is None:
            active_start = time
            active_midis = [midi]
            pending_midis = []
            pending_start = None
            last_voiced_time = time
            previous_time = time
            continue

        current_note = int(np.rint(np.median(active_midis)))
        distance_cents = abs(midi - current_note) * 100.0

        if distance_cents <= cents_tolerance:
            active_midis.append(midi)
            pending_midis = []
            pending_start = None
        else:
            if not pending_midis:
                pending_start = time
            pending_midis.append(midi)

            if len(pending_midis) >= change_frames:
                change_start = (
                    pending_start if pending_start is not None else time
                )
                _finish_note(
                    notes,
                    active_start,
                    change_start,
                    active_midis,
                    min_duration,
                )
                active_start = change_start
                active_midis = pending_midis.copy()
                pending_midis = []
                pending_start = None

        last_voiced_time = time
        previous_time = time

    if active_start is not None and last_voiced_time is not None:
        _finish_note(
            notes,
            active_start,
            last_voiced_time,
            active_midis,
            min_duration,
        )

    return notes


def assign_velocities(
    notes: list[Note],
    samples: np.ndarray,
    sample_rate: int,
    *,
    minimum: int = 40,
    maximum: int = 127,
) -> list[Note]:
    """Assign relative MIDI velocities from each note's RMS amplitude."""
    if not notes or len(samples) == 0:
        return notes
    if minimum < 1 or maximum > 127 or minimum > maximum:
        raise ValueError("velocity range must be within 1..127")

    rms_values: list[float] = []
    for note in notes:
        start = max(0, int(note.start * sample_rate))
        end = min(len(samples), max(start + 1, int(note.end * sample_rate)))
        segment = samples[start:end]
        rms = float(np.sqrt(np.mean(np.square(segment)))) if len(segment) else 0.0
        rms_values.append(rms)

    peak = max(rms_values, default=0.0)
    if peak <= 0:
        return notes

    return [
        Note(
            midi_note=note.midi_note,
            start=note.start,
            end=note.end,
            velocity=int(
                np.clip(
                    round(minimum + (rms / peak) * (maximum - minimum)),
                    minimum,
                    maximum,
                )
            ),
        )
        for note, rms in zip(notes, rms_values)
    ]


def _energy_is_active(
    time: float,
    energy_times: np.ndarray,
    energy_voiced: np.ndarray,
) -> bool:
    if len(energy_times) == 0 or len(energy_voiced) == 0:
        return True
    index = int(np.searchsorted(energy_times, time, side="right") - 1)
    index = max(0, min(index, len(energy_voiced) - 1))
    return bool(energy_voiced[index])


def _median_smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Median-smooth finite pitch values while preserving unvoiced gaps."""
    result = values.copy()
    half = window // 2

    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        finite = values[max(0, i - half):min(len(values), i + half + 1)]
        finite = finite[np.isfinite(finite)]
        if len(finite):
            result[i] = np.median(finite)

    return result


def _finish_note(
    notes: list[Note],
    start: float,
    end: float,
    midis: list[float],
    min_duration: float,
) -> None:
    if end - start < min_duration or not midis:
        return
    midi_note = int(np.clip(np.rint(np.median(midis)), 0, 127))
    notes.append(Note(midi_note=midi_note, start=start, end=end))
