# Audio to Instrument

Windows-first open-source/freemium DAW plugin for converting clean monophonic, single-source melody recordings into musical notes/MIDI, with future support for driving other instrument timbres.

## Development status

**v0.1: Monophonic audio to MIDI research prototype**

Initial development is intentionally focused on the analysis engine before the DAW plugin UI.

## Planned pipeline

Audio → preprocessing → pitch detection → note segmentation → MIDI note conversion → duration/velocity → MIDI export

## Development roadmap

- v0.1: Python research prototype
- v0.2: Accuracy and MIDI improvements
- v0.3: C++/JUCE VST3 plugin
- v0.4: FL Studio/local DAW testing
- v0.5: Basic instrument rendering
- Later: expression, timbre transformation, and polyphonic/source-separation research

## Scope

The first prototype targets clean monophonic, single-source audio such as vocals, humming, whistling, guitar, trumpet, and similar sources. Polyphonic and mixed-source recordings are outside the initial scope.
