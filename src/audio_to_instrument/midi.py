from __future__ import annotations

import mido

from .notes import Note


def export_midi(
    notes: list[Note],
    output_path: str,
    *,
    tempo_bpm: float = 120.0,
) -> None:
    """Write detected notes to a type-1 MIDI file."""
    if tempo_bpm <= 0:
        raise ValueError("tempo_bpm must be positive")

    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(
        mido.MetaMessage(
            "set_tempo",
            tempo=mido.bpm2tempo(tempo_bpm),
            time=0,
        )
    )

    events: list[tuple[float, int, str, int, int]] = []
    for note in notes:
        # Note-off sorts before note-on at the same timestamp so a pitch
        # transition does not briefly create overlapping notes.
        events.append(
            (note.start, 1, "note_on", note.midi_note, note.velocity)
        )
        events.append(
            (note.end, 0, "note_off", note.midi_note, 0)
        )

    events.sort(key=lambda event: (event[0], event[1]))
    last_time = 0.0
    for time_seconds, _, message_type, midi_note, velocity in events:
        delta_seconds = max(0.0, time_seconds - last_time)
        delta_ticks = round(
            mido.second2tick(
                delta_seconds,
                midi.ticks_per_beat,
                mido.bpm2tempo(tempo_bpm),
            )
        )
        track.append(
            mido.Message(
                message_type,
                note=midi_note,
                velocity=velocity,
                time=delta_ticks,
            )
        )
        last_time = time_seconds

    midi.save(output_path)
