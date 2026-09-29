import numpy as np
import pytest

from audio_to_instrument.notes import (
    Note,
    assign_velocities,
    hz_to_midi,
    track_to_notes,
)
from audio_to_instrument.pitch import PitchTrack


def test_a4_is_midi_69():
    assert hz_to_midi(440.0) == pytest.approx(69.0)


def test_a3_is_midi_57():
    assert hz_to_midi(220.0) == pytest.approx(57.0)


def test_zero_frequency_is_unvoiced():
    assert hz_to_midi(0.0) == 0.0


def _track(midis: list[float], step: float = 0.05) -> PitchTrack:
    frequencies = np.array(
        [440.0 * (2.0 ** ((m - 69.0) / 12.0)) for m in midis],
        dtype=np.float32,
    )
    times = np.arange(len(midis), dtype=np.float32) * step
    voiced = np.ones(len(midis), dtype=bool)
    confidence = np.ones(len(midis), dtype=np.float32)
    return PitchTrack(times, frequencies, voiced, confidence)


def test_pitch_jitter_does_not_create_extra_notes():
    track = _track([69.0, 69.2, 68.8, 69.1, 69.0, 69.2])
    notes = track_to_notes(track, min_duration=0.05)
    assert len(notes) == 1
    assert notes[0].midi_note == 69


def test_note_change_requires_persistence():
    track = _track([69.0, 69.0, 71.0, 69.0, 69.0, 69.0, 71.0, 71.0, 71.0])
    notes = track_to_notes(track, min_duration=0.05, change_frames=3)
    assert [n.midi_note for n in notes] == [69, 71]


def test_onset_does_not_split_sustained_same_pitch_note():
    track = _track([60.0] * 12)
    onset_times = np.array([0.10, 0.25, 0.40], dtype=np.float32)
    notes = track_to_notes(
        track,
        onset_times=onset_times,
        min_duration=0.05,
    )
    assert len(notes) == 1
    assert notes[0].midi_note == 60


def test_same_pitch_notes_can_remain_separate_with_energy_gap():
    track = _track([60.0, 60.0, 60.0, 60.0, 60.0, 60.0])
    energy_times = np.arange(6, dtype=np.float32) * 0.05
    energy_voiced = np.array([True, True, False, False, True, True])
    notes = track_to_notes(
        track,
        energy_times=energy_times,
        energy_voiced=energy_voiced,
        min_duration=0.05,
        max_gap=0.06,
    )
    assert len(notes) == 2
    assert notes[0].midi_note == notes[1].midi_note == 60
    assert notes[1].start > notes[0].end


def test_velocity_is_relative_to_note_amplitude():
    samples = np.concatenate(
        [
            np.ones(1000, dtype=np.float32) * 0.2,
            np.ones(1000, dtype=np.float32) * 0.8,
        ]
    )
    notes = [
        Note(60, 0.0, 0.05),
        Note(62, 0.05, 0.10),
    ]
    result = assign_velocities(notes, samples, 20000)
    assert result[1].velocity > result[0].velocity
