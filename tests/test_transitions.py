import numpy as np
import pytest

from audio_to_instrument.articulation import ArticulationFeatures
from audio_to_instrument.notes import Note
from audio_to_instrument.transitions import consolidate_pitch_transitions


def _features(values):
    times = np.arange(len(values), dtype=np.float32) * 0.02
    return ArticulationFeatures(
        times=times,
        rms_db=np.asarray(values, dtype=np.float32),
        onset_strength=np.zeros(len(values), dtype=np.float32),
    )


def test_same_pitch_excursion_collapses_without_release():
    notes = [
        Note(57, 0.00, 0.06),
        Note(58, 0.06, 0.08),
        Note(57, 0.08, 0.14),
    ]
    result = consolidate_pitch_transitions(
        notes,
        _features([-10.0] * 8),
        max_duration=0.04,
        min_neighbor_duration=0.04,
    )
    assert [note.midi_note for note in result] == [57]
    assert result[0].start == pytest.approx(0.00)
    assert result[0].end == pytest.approx(0.14)


def test_monotonic_multi_step_transition_collapses():
    notes = [
        Note(64, 0.00, 0.06),
        Note(63, 0.06, 0.08),
        Note(62, 0.08, 0.10),
        Note(61, 0.10, 0.16),
    ]
    result = consolidate_pitch_transitions(
        notes,
        _features([-10.0] * 9),
        max_duration=0.04,
        min_neighbor_duration=0.04,
    )
    assert [note.midi_note for note in result] == [64, 61]
    assert result[0].end == pytest.approx(result[1].start)


def test_release_protects_same_pitch_excursion():
    notes = [
        Note(57, 0.00, 0.06),
        Note(58, 0.06, 0.08),
        Note(57, 0.08, 0.14),
    ]
    result = consolidate_pitch_transitions(
        notes,
        _features([-10.0, -10.0, -10.0, -10.0, -55.0, -10.0, -10.0, -10.0]),
        max_duration=0.04,
        min_neighbor_duration=0.04,
    )
    assert [note.midi_note for note in result] == [57, 58, 57]
