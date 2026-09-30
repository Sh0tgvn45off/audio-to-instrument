from __future__ import annotations

import numpy as np

from .articulation import ArticulationFeatures
from .notes import Note


def consolidate_pitch_transitions(
    notes: list[Note],
    features: ArticulationFeatures,
    *,
    max_duration: float = 0.10,
    min_neighbor_duration: float = 0.10,
    release_db: float = -50.0,
) -> list[Note]:
    """Collapse short legato pitch-transition runs without removing releases."""
    if max_duration <= 0:
        raise ValueError("max_duration must be positive")
    if min_neighbor_duration < 0:
        raise ValueError("min_neighbor_duration cannot be negative")
    if len(notes) < 3 or len(features.times) == 0:
        return notes

    result = list(notes)
    index = 1

    while index < len(result) - 1:
        previous = result[index - 1]
        middle = result[index]

        if middle.duration > max_duration:
            index += 1
            continue
        if previous.duration < min_neighbor_duration:
            index += 1
            continue

        end_index = index
        while (
            end_index + 1 < len(result) - 1
            and result[end_index + 1].duration <= max_duration
        ):
            end_index += 1

        following = result[end_index + 1]
        if following.duration < min_neighbor_duration:
            index += 1
            continue

        previous_pitch = previous.midi_note
        following_pitch = following.midi_note
        pitches = [note.midi_note for note in result[index:end_index + 1]]

        if previous_pitch == following_pitch:
            candidate = all(pitch != previous_pitch for pitch in pitches)
        else:
            low = min(previous_pitch, following_pitch)
            high = max(previous_pitch, following_pitch)
            candidate = all(low < pitch < high for pitch in pitches)

            if candidate:
                direction = 1 if following_pitch > previous_pitch else -1
                candidate = all(
                    (right - left) * direction > 0
                    for left, right in zip(pitches, pitches[1:])
                )
                candidate = candidate and (
                    (pitches[0] - previous_pitch) * direction > 0
                    and (following_pitch - pitches[-1]) * direction > 0
                )

        if not candidate:
            index += 1
            continue

        boundary_window = min(
            max_duration,
            max(features.times[1] - features.times[0], 0.01)
            if len(features.times) > 1
            else 0.01,
        )
        boundaries = [
            result[position].start
            for position in range(index, end_index + 1)
        ]
        boundaries.extend(
            result[position].end
            for position in range(index, end_index + 1)
        )

        if any(
            _has_release(
                boundary - boundary_window,
                boundary + boundary_window,
                features,
                release_db=release_db,
            )
            for boundary in boundaries
        ):
            index += 1
            continue

        result[index - 1] = Note(
            midi_note=previous.midi_note,
            start=previous.start,
            end=following.start,
            velocity=previous.velocity,
        )
        result[end_index + 1] = Note(
            midi_note=following.midi_note,
            start=following.start,
            end=following.end,
            velocity=following.velocity,
        )
        del result[index:end_index + 1]
        index = max(1, index - 1)

    return result


def _has_release(
    start: float,
    end: float,
    features: ArticulationFeatures,
    *,
    release_db: float,
) -> bool:
    if len(features.times) == 0 or end < start:
        return False

    epsilon = 1e-6
    mask = (
        (features.times >= start - epsilon)
        & (features.times <= end + epsilon)
    )
    return bool(np.any(features.rms_db[mask] <= release_db))
