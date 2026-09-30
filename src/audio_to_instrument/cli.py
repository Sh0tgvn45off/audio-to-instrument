from __future__ import annotations

import argparse
import csv

from .audio import load_audio, preprocess
from .articulation import detect_articulation_features
from .midi import export_midi
from .notes import (
    assign_velocities,
    detect_energy_regions,
    track_to_notes,
)
from .pitch import detect_pitch


def write_report(path: str, notes) -> None:
    """Write detected notes to a simple CSV for manual evaluation."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["midi_note", "start", "end", "duration", "velocity"])
        for note in notes:
            writer.writerow(
                [
                    note.midi_note,
                    f"{note.start:.6f}",
                    f"{note.end:.6f}",
                    f"{note.duration:.6f}",
                    note.velocity,
                ]
            )


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
        default=0.05,
        help="Minimum note duration in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--energy-top-db",
        type=float,
        default=35.0,
        help="Legacy energy-mask threshold in dB; articulation-aware mode uses continuous RMS evidence (default: 35)",
    )
    parser.add_argument(
        "--energy-release-frames",
        type=int,
        default=3,
        help="Consecutive low-energy frames required before release evidence is considered (default: 3)",
    )
    parser.add_argument(
        "--transition-max-duration",
        type=float,
        default=0.10,
        help="Maximum duration of a short intermediate pitch region that may be treated as a legato transition (default: 0.10)",
    )
    parser.add_argument(
        "--transition-min-neighbor-duration",
        type=float,
        default=0.10,
        help="Minimum duration required for neighboring notes before a short transition region can be collapsed (default: 0.10)",
    )
    parser.add_argument(
        "--report",
        help="Optional CSV path for detailed note timing/velocity output",
    )
    args = parser.parse_args()

    audio = load_audio(args.input)
    samples = preprocess(audio.samples)

    pitch = detect_pitch(
        samples,
        audio.sample_rate,
        confidence_threshold=args.confidence,
    )

    articulation = detect_articulation_features(
        samples,
        audio.sample_rate,
    )

    notes = track_to_notes(
        pitch,
        articulation_features=articulation,
        min_duration=args.min_duration,
        energy_release_frames=args.energy_release_frames,
        transition_max_duration=args.transition_max_duration,
        transition_min_neighbor_duration=args.transition_min_neighbor_duration,
    )
    notes = assign_velocities(notes, samples, audio.sample_rate)

    export_midi(notes, args.output)

    if args.report:
        write_report(args.report, notes)

    print(f"Detected {len(notes)} notes")
    for note in notes:
        print(
            f"  MIDI {note.midi_note:3d} | "
            f"{note.start:7.3f}s - {note.end:7.3f}s | "
            f"duration {note.duration:6.3f}s | "
            f"velocity {note.velocity:3d}"
        )
    print(f"Wrote MIDI: {args.output}")
    if args.report:
        print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
