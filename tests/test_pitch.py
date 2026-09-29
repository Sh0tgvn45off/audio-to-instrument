import pytest

from audio_to_instrument.notes import hz_to_midi


def test_a4_is_midi_69():
    assert hz_to_midi(440.0) == pytest.approx(69.0)


def test_a3_is_midi_57():
    assert hz_to_midi(220.0) == pytest.approx(57.0)


def test_zero_frequency_is_unvoiced():
    assert hz_to_midi(0.0) == 0.0
