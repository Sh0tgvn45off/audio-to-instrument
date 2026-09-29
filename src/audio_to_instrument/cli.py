from __future__ import annotations

import argparse

from .audio import load_audio, preprocess
from .midi import export_midi
from .notes import track_to_notes
from .pitch import detect_pitch


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert clean monophonic audio to MIDI.")
    parser.add_argument("input", help="Input WAV/audio file")
    parser.add_argument("output", help="Output MIDI file")
    args = parser.parse_args()

    audio = load_audio(args.input)
    samples = preprocess(audio.samples)
    pitch = detect_pitch(samples, audio.sample_rate)
    notes = track_to_notes(pitch)
    export_midi(notes, args.output)

    print(f"Detected {len(notes)} notes")
    print(f"Wrote MIDI: {args.output}")


if __name__ == "__main__":
    main()
