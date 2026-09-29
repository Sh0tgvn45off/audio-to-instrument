from __future__ import annotations

import argparse

from .audio import load_audio, preprocess
from .midi import export_midi
from .notes import assign_velocities, detect_onsets, track_to_notes
from .pitch import detect_pitch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert clean monophonic audio to MIDI."
    )
    parser.add_argument("input", help="Input WAV/audio file")
    parser.add_argument("output", help="Output MIDI file")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.65,
        help="Minimum pYIN confidence for a voiced frame (default: 0.65)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.08,
        help="Minimum note duration in seconds (default: 0.08)",
    )
    parser.add_argument(
        "--no-onsets",
        action="store_true",
        help="Disable onset detection for repeated notes at the same pitch",
    )
    args = parser.parse_args()

    audio = load_audio(args.input)
    samples = preprocess(audio.samples)

    pitch = detect_pitch(
        samples,
        audio.sample_rate,
        confidence_threshold=args.confidence,
    )

    onset_times = None
    if not args.no_onsets:
        onset_times = detect_onsets(samples, audio.sample_rate)

    notes = track_to_notes(
        pitch,
        onset_times=onset_times,
        min_duration=args.min_duration,
    )
    notes = assign_velocities(notes, samples, audio.sample_rate)

    export_midi(notes, args.output)

    print(f"Detected {len(notes)} notes")
    for note in notes:
        print(
            f"  MIDI {note.midi_note:3d} | "
            f"{note.start:7.3f}s - {note.end:7.3f}s | "
            f"velocity {note.velocity:3d}"
        )
    print(f"Wrote MIDI: {args.output}")


if __name__ == "__main__":
    main()
