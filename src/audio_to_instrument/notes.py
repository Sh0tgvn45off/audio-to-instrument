from __future__ import annotations

from dataclasses import dataclass

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
    if frequency_hz <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(frequency_hz / 440.0)


def track_to_notes(
    track: PitchTrack,
    *,
    min_duration: float = 0.08,
    max_gap: float = 0.06,
    cents_tolerance: float = 80.0,
) -> list[Note]:
    """Convert a monophonic pitch track into coarse MIDI notes.

    This is deliberately a simple v0.1 segmenter. Accuracy improvements will
    be added after real recordings are evaluated.
    """
    notes: list[Note] = []
    active_start: float | None = None
    active_midis: list[float] = []
    last_voiced_time: float | None = None

    for time, frequency, is_voiced in zip(track.times, track.frequencies_hz, track.voiced):
        midi = hz_to_midi(float(frequency)) if is_voiced else 0.0

        if not is_voiced:
            if active_start is not None and last_voiced_time is not None:
                if float(time - last_voiced_time) > max_gap:
                    _finish_note(notes, active_start, float(last_voiced_time), active_midis, min_duration)
                    active_start = None
                    active_midis = []
            continue

        if active_start is None:
            active_start = float(time)
            active_midis = [midi]
            last_voiced_time = float(time)
            continue

        median_midi = float(np.median(active_midis))
        if abs(midi - median_midi) * 100.0 > cents_tolerance:
            _finish_note(notes, active_start, float(last_voiced_time), active_midis, min_duration)
            active_start = float(time)
            active_midis = [midi]
        else:
            active_midis.append(midi)

        last_voiced_time = float(time)

    if active_start is not None and last_voiced_time is not None:
        _finish_note(notes, active_start, float(last_voiced_time), active_midis, min_duration)

    return notes


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
