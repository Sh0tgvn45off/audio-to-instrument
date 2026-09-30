# Testing Guide

## Current status

The project is a **v0.1.x Python monophonic audio-to-MIDI research prototype**.

The current development target is articulation-aware note segmentation.

**Current milestone: `v0.1.3` | Phase: Articulation Hysteresis & Release Detection** The analyzer is being validated locally before the project moves to the production C++/JUCE/VST3 implementation.

There is **no graphical UI yet**. Testing is done from the VS Code terminal.

## Requirements

- Windows
- Python 3.11
- Git
- VS Code
- VS Code Python extension
- pip and Python `venv`
- A clean monophonic WAV recording

Recommended test material is a single source such as:

- singing
- humming
- whistling
- single-note guitar
- trumpet
- another isolated melodic instrument

Do not use full songs, chords, mixed vocals/instruments, or other polyphonic material for the current milestone.

## Setup

Clone the repository:

```powershell
git clone https://github.com/Sh0tgvn45off/audio-to-instrument.git
cd audio-to-instrument
git checkout development
```

Create the virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Verify the environment:

```powershell
python --version
pytest
```

Python 3.11 is the current CI target.

## Automated tests

Run all tests:

```powershell
pytest
```

The regression suite currently checks:

- frequency-to-MIDI conversion
- pitch jitter not creating extra notes
- persistent pitch changes
- onset detections not fragmenting sustained notes
- repeated same-pitch notes separated by an energy gap
- short energy dips not splitting sustained notes
- confirmed energy releases ending notes after hysteresis
- relative velocity estimation

All tests must pass before considering a development change for merge.

## WAV to MIDI test

Put a test WAV somewhere accessible from the repository, for example:

```text
test_audio/
    AUDIE_TEST_HAM_0.wav
```

Run:

```powershell
audio-to-midi test_audio\AUDIE_TEST_HAM_0.wav test_audio\AUDIE_TEST_HAM_0.mid
```

For a detailed note report:

```powershell
audio-to-midi test_audio\AUDIE_TEST_HAM_0.wav test_audio\AUDIE_TEST_HAM_0.mid --report test_audio\AUDIE_TEST_HAM_0_notes.csv
```

The CSV contains:

```text
midi_note,start,end,duration,velocity
```

The command also prints the same information in the terminal.

## Parameters for experiments

Confidence threshold:

```powershell
audio-to-midi input.wav output.mid --confidence 0.70
```

Minimum note duration:

```powershell
audio-to-midi input.wav output.mid --min-duration 0.08
```

Energy threshold:

```powershell
audio-to-midi input.wav output.mid --energy-top-db 35
```

The current segmentation decisions are persistent pitch changes and energy releases. Release confirmation uses consecutive inactive energy frames (`--energy-release-frames`) so brief envelope dips do not terminate sustained notes. Onset detection remains an isolated research utility and is not currently allowed to create MIDI note boundaries because that caused over-segmentation of sustained notes.

## Manual MIDI validation

Open the generated MIDI in a DAW such as FL Studio and use a simple piano or other neutral instrument.

Compare the MIDI against the original WAV.

Check these separately:

### 1. Pitch

- Are the notes the correct musical pitches?
- Are octave errors present?
- Are short notes more likely to be misidentified?

### 2. Note starts

- Does each note begin at approximately the correct time?
- Are attacks delayed?
- Are false notes created by consonants, breaths, or transients?

### 3. Note ends

- Does a sustained note remain sustained?
- Does the note stop when the original sound releases?
- Are releases being cut too early?

### 4. Articulation

Specifically test:

- sustained notes
- staccato notes
- repeated notes at the same pitch
- rapid notes
- notes with short gaps
- pitch changes with no silence
- vibrato or small pitch fluctuations

The current goal is for a repeated same-pitch staccato performance to become separate MIDI notes while a sustained note remains one MIDI note.

### 5. Velocity

Check whether louder notes generally produce higher MIDI velocities.

## Current known limitations

- Monophonic material only
- Clean isolated source preferred
- No polyphonic transcription
- No source separation
- No graphical UI
- Pitch bends and vibrato are not yet exported as continuous MIDI expression
- Articulation detection is still experimental
- Release hysteresis is currently frame-based and will need validation against more recordings
- MIDI timing is frame-based and will not yet be sample-accurate
- Python is the research/prototype engine, not the final DAW plugin implementation

## Milestone gate: `v0.1.3` | Phase: Articulation Hysteresis & Release Detection

Before this milestone is considered complete, verify that:

1. Short energy dips do not split sustained notes.
2. Genuine repeated-note gaps still produce separate MIDI notes.
3. Release timing is audibly closer to the source than the previous single-frame energy decision.
4. The existing pitch and onset regression protections remain intact.

## Merge gate

Do **not** merge the development branch into `main` until:

1. GitHub Actions passes.
2. Local `pytest` passes.
3. The reference WAV produces sensible pitches.
4. Sustained notes are not fragmented by false onset detections.
5. Repeated same-pitch staccato notes remain separate.
6. Note starts and releases are audibly reasonable.
7. No obvious octave or persistent pitch errors are introduced.
8. Manual MIDI inspection in a DAW is acceptable.
9. The change does not regress existing tests.
10. The development branch has been reviewed.

The stable `main` branch should remain unchanged while articulation work is being evaluated.
