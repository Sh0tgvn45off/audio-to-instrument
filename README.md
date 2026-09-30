# Audio to Instrument

Windows-first open-source/freemium DAW plugin for converting clean monophonic, single-source melody recordings into musical notes/MIDI, with future support for driving other instrument timbres.

## Development status

**v0.1.x: Monophonic audio to MIDI research prototype**

**Current milestone: `v0.1.3` | Phase: Articulation Hysteresis & Release Detection**

The current prototype is a Python analysis engine. It is intentionally being developed and evaluated before moving the stable engine into C++/JUCE/VST3.

The current priority is **transcription quality**, especially note boundaries and articulation. UI work is deferred until the analysis engine is reliable enough to validate in a DAW.

## Current pipeline

Audio
→ mono loading
→ silence trimming/normalization
→ pYIN fundamental-frequency detection
→ confidence filtering
→ pitch smoothing
→ local energy envelope
→ release hysteresis
→ energy-aware note segmentation
→ persistent pitch-change detection
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
    pip install -e .

Convert an audio recording:

    audio-to-midi input.wav output.mid

Useful options:

    audio-to-midi input.wav output.mid --confidence 0.70
    audio-to-midi input.wav output.mid --min-duration 0.08
    audio-to-midi input.wav output.mid --energy-top-db 35
    audio-to-midi input.wav output.mid --report notes.csv

The command prints detected MIDI note, start/end time, duration, and estimated velocity. The optional CSV report makes manual transcription analysis easier.

## Development roadmap

- **v0.1:** Python monophonic audio-to-MIDI prototype
- **v0.1.3 | Phase: Articulation Hysteresis & Release Detection:** separate attack evidence from release confirmation, prevent short energy dips from fragmenting sustained notes, and improve repeated-note separation
- **v0.1.x:** evaluation against real recordings and accuracy improvements
- **v0.2:** stronger note-boundary handling, pitch-bend/expression research, and better MIDI export
- **v0.3:** C++/JUCE VST3 plugin
- **v0.4:** FL Studio and other DAW testing
- **v0.5:** basic instrument rendering
- **Later:** timbre transformation, AI-assisted rendering, and polyphonic/source-separation research

## Design principle

The analysis engine comes first. The eventual plugin should expose the same core pipeline inside a real-time/offline-capable DAW environment without coupling the signal-processing algorithms to the UI.

## Testing

See [TESTING.md](TESTING.md) for the complete Windows/VS Code setup, automated tests, WAV-to-MIDI workflow, manual DAW validation checklist, and merge criteria.

## License

See `LICENSE`.


## Current development milestone

**v0.1.4 | Articulation Classification & Note Boundary Scoring**

This milestone adds articulation-aware segmentation for clean monophonic singing/humming. Continuous RMS energy, pYIN confidence, and spectral-flux onset evidence are combined so short energy dips do not automatically become note-offs. Onset evidence remains supporting evidence rather than a hard boundary.
