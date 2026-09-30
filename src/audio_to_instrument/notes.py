from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .articulation import ArticulationFeatures, sample_articulation_features
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
    return times.astype(np.float32), rms >= threshold


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
    energy_release_frames: int = 3,
    articulation_features: ArticulationFeatures | None = None,
    release_db: float = -50.0,
    hard_release_db: float = -60.0,
    release_confidence: float = 0.45,
    hard_release_frames: int = 3,
    transition_max_duration: float = 0.10,
    transition_min_neighbor_duration: float = 0.10,
) -> list[Note]:
    """Convert a monophonic pitch track into articulation-aware MIDI notes.

    Pitch determines note identity. Persistent pitch changes determine
    boundaries between different notes. With articulation features supplied,
    RMS level, pYIN confidence, and spectral-flux onset evidence jointly
    determine whether a release is real. A short energy dip is not enough to
    end a note.

    The legacy energy-mask path remains available for deterministic tests and
    backwards compatibility. Production callers should provide
    ``articulation_features`` so energy is treated as evidence rather than a
    hard voiced/silent switch.

    Onset detection is supporting evidence only. It is never a hard boundary
    because singing onsets can be soft and spectral transients can occur
    inside sustained notes.
    """
    if change_frames < 1:
        raise ValueError("change_frames must be at least 1")
    if min_duration < 0:
        raise ValueError("min_duration cannot be negative")
    if energy_release_frames < 1:
        raise ValueError("energy_release_frames must be at least 1")
    if hard_release_frames < 1:
        raise ValueError("hard_release_frames must be at least 1")
    if release_confidence < 0.0 or release_confidence > 1.0:
        raise ValueError("release_confidence must be between 0 and 1")
    if hard_release_db > release_db:
        raise ValueError("hard_release_db must be at or below release_db")
    if transition_max_duration <= 0:
        raise ValueError("transition_max_duration must be positive")
    if transition_min_neighbor_duration < 0:
        raise ValueError("transition_min_neighbor_duration cannot be negative")

    if len(track.times) == 0:
        return []

    raw_midi = np.array(
        [
            hz_to_midi(float(frequency)) if voiced else np.nan
            for frequency, voiced in zip(
                track.frequencies_hz, track.voiced
            )
        ],
        dtype=np.float32,
    )
    smoothed = _median_smooth(raw_midi, window=5)

    # Onsets remain available for future attack-time refinement. They are not
    # boundaries by themselves because that caused severe over-segmentation.
    _ = onset_times

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
    inactive_energy_frames = 0
    hard_inactive_frames = 0
    first_inactive_energy_time: float | None = None

    use_articulation = articulation_features is not None

    for time, value, is_voiced, confidence in zip(
        track.times, smoothed, track.voiced, track.confidence
    ):
        time = float(time)
        confidence = float(confidence)

        if use_articulation:
            rms_db, onset_strength = sample_articulation_features(
                time, articulation_features
            )
            low_energy = rms_db <= release_db
            hard_silence = rms_db <= hard_release_db
            energy_active = not low_energy
        else:
            onset_strength = 0.0
            energy_active = _energy_is_active(
                time, energy_times, energy_voiced
            )
            low_energy = not energy_active
            hard_silence = not energy_active

        if not is_voiced or not np.isfinite(value):
            if active_start is not None and last_voiced_time is not None:
                gap = time - last_voiced_time
                if low_energy:
                    inactive_energy_frames += 1
                    hard_inactive_frames += 1 if hard_silence else 0
                    if first_inactive_energy_time is None:
                        first_inactive_energy_time = time
                else:
                    inactive_energy_frames = 0
                    hard_inactive_frames = 0
                    first_inactive_energy_time = None

                release_confirmed = (
                    inactive_energy_frames >= energy_release_frames
                    and (
                        confidence <= release_confidence
                        or hard_inactive_frames >= hard_release_frames
                    )
                )

                if gap > max_gap or release_confirmed:
                    end = (
                        first_inactive_energy_time
                        if release_confirmed
                        and first_inactive_energy_time is not None
                        else (
                            (last_voiced_time + time) / 2.0
                            if low_energy
                            else last_voiced_time
                        )
                    )
                    _finish_note(
                        notes,
                        active_start,
                        end,
                        active_midis,
                        min_duration,
                    )
                    active_start = None
                    active_midis = []
                    pending_midis = []
                    pending_start = None
                    inactive_energy_frames = 0
                    hard_inactive_frames = 0
                    first_inactive_energy_time = None
            continue

        if use_articulation:
            if low_energy:
                inactive_energy_frames += 1
                hard_inactive_frames += 1 if hard_silence else 0
                if first_inactive_energy_time is None:
                    first_inactive_energy_time = time

                # A strong recovery/attack while confidence remains healthy
                # cancels a possible release. This protects sustained humming
                # from short envelope dips and transient spectral changes.
                if onset_strength >= 0.5 and confidence > release_confidence:
                    inactive_energy_frames = 0
                    hard_inactive_frames = 0
                    first_inactive_energy_time = None
                else:
                    release_confirmed = (
                        inactive_energy_frames >= energy_release_frames
                        and (
                            confidence <= release_confidence
                            or hard_inactive_frames >= hard_release_frames
                        )
                    )
                    if release_confirmed and active_start is not None:
                        end = (
                            first_inactive_energy_time
                            if first_inactive_energy_time is not None
                            else time
                        )
                        _finish_note(
                            notes,
                            active_start,
                            end,
                            active_midis,
                            min_duration,
                        )
                        active_start = None
                        active_midis = []
                        pending_midis = []
                        pending_start = None
                        inactive_energy_frames = 0
                        hard_inactive_frames = 0
                        first_inactive_energy_time = None
                        last_voiced_time = None
                        continue
            else:
                inactive_energy_frames = 0
                hard_inactive_frames = 0
                first_inactive_energy_time = None
        else:
            if not energy_active:
                inactive_energy_frames += 1
                if first_inactive_energy_time is None:
                    first_inactive_energy_time = time
                if inactive_energy_frames < energy_release_frames:
                    continue

                if active_start is not None:
                    _finish_note(
                        notes,
                        active_start,
                        first_inactive_energy_time,
                        active_midis,
                        min_duration,
                    )
                    active_start = None
                    active_midis = []
                    pending_midis = []
                    pending_start = None
                    inactive_energy_frames = 0
                    first_inactive_energy_time = None
                    last_voiced_time = None
                    continue
            else:
                inactive_energy_frames = 0
                first_inactive_energy_time = None

        midi = float(value)

        if active_start is None:
            active_start = time
            active_midis = [midi]
            pending_midis = []
            pending_start = None
            last_voiced_time = time
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

    if active_start is not None and last_voiced_time is not None:
        _finish_note(
            notes,
            active_start,
            last_voiced_time,
            active_midis,
            min_duration,
        )

    if use_articulation:
        notes = _consolidate_pitch_transitions(
            notes,
            articulation_features,
            max_duration=transition_max_duration,
            min_neighbor_duration=transition_min_neighbor_duration,
            release_db=release_db,
        )

    return notes


def _consolidate_pitch_transitions(
    notes: list[Note],
    features: ArticulationFeatures,
    *,
    max_duration: float,
    min_neighbor_duration: float,
    release_db: float,
) -> list[Note]:
    """Remove short legato transition regions mistaken for musical notes.

    Singing F0 often passes through intermediate pitches during a transition.
    Research on singing transcription treats these regions as transitions
    rather than independent notes, using pitch-trajectory stability and
    explicit transition modeling. We therefore suppress a short middle note
    when its pitch lies between two stable neighboring notes and there is no
    release evidence at either boundary.

    Energy/release evidence is deliberately required to be absent before a
    transition is collapsed. This protects genuine short articulated notes.
    """
    if len(notes) < 3 or len(features.times) == 0:
        return notes

    result = list(notes)
    changed = True

    while changed and len(result) >= 3:
        changed = False

        for index in range(1, len(result) - 1):
            previous = result[index - 1]
            middle = result[index]
            following = result[index + 1]

            if middle.duration > max_duration:
                continue
            if previous.duration < min_neighbor_duration:
                continue
            if following.duration < min_neighbor_duration:
                continue

            previous_pitch = previous.midi_note
            middle_pitch = middle.midi_note
            following_pitch = following.midi_note

            # The short region must sit between the neighboring pitches.
            if previous_pitch == following_pitch:
                continue
            low = min(previous_pitch, following_pitch)
            high = max(previous_pitch, following_pitch)
            if not low < middle_pitch < high:
                continue

            # A genuine articulation/release is evidence for a real note
            # boundary, so do not collapse this candidate.
            # Only inspect the actual transition interval. A release that
            # occurs well before the transition can belong to the preceding
            # note and must not classify an otherwise legato pitch transition.
            if _has_release_in_interval(
                middle.start,
                middle.end,
                features,
                release_db=release_db,
            ):
                continue

            # Removing the transition region must not manufacture a gap.
            result[index - 1] = Note(
                midi_note=previous.midi_note,
                start=previous.start,
                end=following.start,
                velocity=previous.velocity,
            )
            result[index + 1] = Note(
                midi_note=following.midi_note,
                start=following.start,
                end=following.end,
                velocity=following.velocity,
            )
            del result[index]
            changed = True
            break

    return result


def _has_release_in_interval(
    start: float,
    end: float,
    features: ArticulationFeatures,
    *,
    release_db: float,
) -> bool:
    """Return whether release evidence occurs inside a candidate transition."""
    if len(features.times) == 0 or end < start:
        return False

    epsilon = 1e-6
    mask = (
        (features.times >= start - epsilon)
        & (features.times <= end + epsilon)
    )
    if not np.any(mask):
        return False

    return bool(np.any(features.rms_db[mask] <= release_db))


def assign_velocities(
    notes: list[Note],
    samples: np.ndarray,
    sample_rate: int,
    *,
    minimum: int = 40,
    maximum: int = 127,
    attack_window: float = 0.08,
) -> list[Note]:
    """Assign MIDI velocity from the RMS level near each note attack.

    MIDI velocity represents attack intensity more closely than average note
    loudness, so a short leading window is used instead of the whole note.
    """
    if not notes or len(samples) == 0:
        return notes
    if minimum < 1 or maximum > 127 or minimum > maximum:
        raise ValueError("velocity range must be within 1..127")
    if attack_window <= 0:
        raise ValueError("attack_window must be positive")

    rms_values: list[float] = []
    for note in notes:
        start = max(0, int(note.start * sample_rate))
        attack_end = min(
            len(samples),
            max(start + 1, int((note.start + attack_window) * sample_rate)),
        )
        end = min(
            len(samples),
            max(start + 1, int(note.end * sample_rate)),
        )
        end = min(end, attack_end)
        segment = samples[start:end]
        rms = (
            float(np.sqrt(np.mean(np.square(segment))))
            if len(segment)
            else 0.0
        )
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
