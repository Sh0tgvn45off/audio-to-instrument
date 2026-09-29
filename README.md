# Audio to Instrument

Windows-first open-source/freemium DAW plugin for converting clean monophonic, single-source melody recordings into musical notes/MIDI, with future support for driving other instrument timbres.

## Development status

**v0.1: Monophonic audio to MIDI research prototype**

The current prototype is a Python analysis engine. It is intentionally being developed and evaluated before moving the stable engine into C++/JUCE/VST3.

## Current pipeline

Audio
→ mono loading
→ silence trimming/normalization
→ pYIN fundamental-frequency detection
→ confidence filtering
→ pitch smoothing
→ onset detection
→ note segmentation with change hysteresis
→ MIDI note conversion
→ RMS-based velocity estimation
→ MIDI export

## Supported input concept

The first prototype targets clean monophonic, single-source material such as:

- singing
- humming
- whistling
- single-note guitar
- trumpet
- similar melodic instruments

Polyphonic recordings, chords, full mixes, and source separation are outside the initial scope.

## Run the prototype

Create and activate a Python virtual environment, then install dependencies:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

Convert an audio recording:

    audio-to-midi input.wav output.mid

Useful options:

    audio-to-midi input.wav output.mid --confidence 0.70
    audio-to-midi input.wav output.mid --min-duration 0.10
    audio-to-midi input.wav output.mid --no-onsets

The command prints the detected MIDI note, start/end time, and estimated velocity for each note.

## Development roadmap

- **v0.1:** Python monophonic audio-to-MIDI prototype
- **v0.1.x:** evaluation against real recordings and accuracy improvements
- **v0.2:** stronger onset/note-boundary handling, pitch-bend/expression research, and better MIDI export
- **v0.3:** C++/JUCE VST3 plugin
- **v0.4:** FL Studio and other DAW testing
- **v0.5:** basic instrument rendering
- **Later:** timbre transformation, AI-assisted rendering, and polyphonic/source-separation research

## Design principle

The analysis engine comes first. The eventual plugin should expose the same core pipeline inside a real-time/offline-capable DAW environment without coupling the signal-processing algorithms to the UI.

## License

See `LICENSE`.