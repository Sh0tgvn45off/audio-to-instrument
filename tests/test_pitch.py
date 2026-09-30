import numpy as np
import pytest
import mido

from audio_to_instrument.articulation import ArticulationFeatures
from audio_to_instrument.midi import export_midi
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
    track = _track(
        [69.0, 69.0, 71.0, 69.0, 69.0, 69.0, 71.0, 71.0, 71.0]
    )
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



def test_short_energy_dip_does_not_split_sustained_note():
    track = _track([60.0] * 8)
    energy_times = np.arange(8, dtype=np.float32) * 0.05
    energy_voiced = np.array(
        [True, True, False, True, True, True, True, True]
    )
    notes = track_to_notes(
        track,
        energy_times=energy_times,
        energy_voiced=energy_voiced,
        min_duration=0.05,
        max_gap=0.20,
        energy_release_frames=2,
    )
    assert len(notes) == 1
    assert notes[0].midi_note == 60


def test_sustained_energy_release_ends_note_after_hysteresis():
    track = _track([60.0] * 8)
    energy_times = np.arange(8, dtype=np.float32) * 0.05
    energy_voiced = np.array(
        [True, True, False, False, False, True, True, True]
    )
    notes = track_to_notes(
        track,
        energy_times=energy_times,
        energy_voiced=energy_voiced,
        min_duration=0.05,
        max_gap=0.20,
        energy_release_frames=3,
    )
    assert len(notes) == 2
    assert notes[0].midi_note == notes[1].midi_note == 60
    assert notes[1].start > notes[0].end


def test_articulation_energy_dip_does_not_split_sustained_note():
    track = _track([60.0] * 8)
    times = np.arange(8, dtype=np.float32) * 0.05
    features = ArticulationFeatures(
        times=times,
        rms_db=np.array(
            [-10.0, -10.0, -55.0, -10.0, -10.0, -10.0, -10.0, -10.0],
            dtype=np.float32,
        ),
        onset_strength=np.zeros(8, dtype=np.float32),
    )
    notes = track_to_notes(
        track,
        articulation_features=features,
        min_duration=0.05,
        energy_release_frames=3,
    )
    assert len(notes) == 1
    assert notes[0].midi_note == 60


def test_articulation_requires_real_release_evidence():
    track = _track([60.0] * 9)
    times = np.arange(9, dtype=np.float32) * 0.05
    features = ArticulationFeatures(
        times=times,
        rms_db=np.array(
            [-10.0, -10.0, -70.0, -70.0, -70.0, -10.0, -10.0, -10.0, -10.0],
            dtype=np.float32,
        ),
        onset_strength=np.zeros(9, dtype=np.float32),
    )
    notes = track_to_notes(
        track,
        articulation_features=features,
        min_duration=0.05,
        energy_release_frames=3,
        hard_release_frames=3,
    )
    assert len(notes) == 2
    assert notes[0].midi_note == notes[1].midi_note == 60
    assert notes[1].start > notes[0].end

def test_short_monotonic_pitch_transition_is_not_a_note():
    track = _track(
        [57.0, 57.0, 57.0, 57.0, 56.0, 55.0, 55.0, 55.0, 55.0],
        step=0.02,
    )
    times = np.arange(9, dtype=np.float32) * 0.02
    features = ArticulationFeatures(
        times=times,
        rms_db=np.full(9, -10.0, dtype=np.float32),
        onset_strength=np.zeros(9, dtype=np.float32),
    )

    notes = track_to_notes(
        track,
        articulation_features=features,
        min_duration=0.02,
        change_frames=1,
        transition_max_duration=0.04,
        transition_min_neighbor_duration=0.05,
    )

    assert [note.midi_note for note in notes] == [57, 55]
    assert notes[0].end == pytest.approx(notes[1].start)


def test_short_pitch_transition_with_release_is_preserved():
    track = _track(
        [57.0, 57.0, 57.0, 56.0, 55.0, 55.0, 55.0],
        step=0.02,
    )
    times = np.arange(7, dtype=np.float32) * 0.02
    features = ArticulationFeatures(
        times=times,
        rms_db=np.array(
            [-10.0, -10.0, -55.0, -10.0, -10.0, -10.0, -10.0],
            dtype=np.float32,
        ),
        onset_strength=np.zeros(7, dtype=np.float32),
    )

    notes = track_to_notes(
        track,
        articulation_features=features,
        min_duration=0.02,
        change_frames=1,
        transition_max_duration=0.04,
        transition_min_neighbor_duration=0.04,
    )

    assert [note.midi_note for note in notes] == [57, 56, 55]


def test_velocity_is_relative_to_attack_amplitude():
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
    result = assign_velocities(
        notes,
        samples,
        20000,
        attack_window=0.03,
    )
    assert result[1].velocity > result[0].velocity


def test_midi_uses_note_velocity_and_closes_before_new_pitch(tmp_path):
    notes = [
        Note(60, 0.0, 0.5, velocity=70),
        Note(64, 0.5, 1.0, velocity=110),
    ]
    output = tmp_path / "test_velocity_and_order.mid"
    export_midi(notes, str(output))

    midi = mido.MidiFile(output)
    messages = [
        message
        for track in midi.tracks
        for message in track
        if message.type in {"note_on", "note_off"}
    ]

    assert messages[0].type == "note_on"
    assert messages[0].note == 60
    assert messages[0].velocity == 70
    assert messages[1].type == "note_off"
    assert messages[1].note == 60
    assert messages[2].type == "note_on"
    assert messages[2].note == 64
    assert messages[2].velocity == 110


def test_preprocess_preserves_timeline_by_default():
    from audio_to_instrument.audio import preprocess

    samples = np.zeros(1000, dtype=np.float32)
    samples[200:800] = 0.5
    result = preprocess(samples)

    assert len(result) == len(samples)
    assert np.argmax(result) == np.argmax(samples)
    assert result[200] == pytest.approx(1.0)
