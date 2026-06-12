"""
Step 1 & 2: Collect/Generate MIDI Music Data and Preprocess
- Generates synthetic classical-style MIDI sequences
- Preprocesses into note sequences using music21
"""

import os
import random
import numpy as np
from midiutil import MIDIFile
from music21 import converter, instrument, note, chord, stream

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

OUTPUT_DIR = "midi_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Musical building blocks ───────────────────────────────────────────────────
SCALES = {
    "C_major":  [60, 62, 64, 65, 67, 69, 71, 72],
    "G_major":  [55, 57, 59, 60, 62, 64, 66, 67],
    "A_minor":  [57, 59, 60, 62, 64, 65, 67, 69],
    "D_major":  [62, 64, 66, 67, 69, 71, 73, 74],
    "F_major":  [53, 55, 57, 58, 60, 62, 64, 65],
}
CHORDS = {
    "C_major":  [[60,64,67],[65,69,72],[67,71,74],[62,65,69]],
    "A_minor":  [[57,60,64],[62,65,69],[64,67,71],[60,64,67]],
}
DURATIONS = [0.25, 0.5, 1.0]


def generate_melody(scale_notes, length=32):
    """Generate a simple melodic phrase using the scale."""
    melody = []
    prev = random.choice(scale_notes[2:6])          # start near the middle
    for _ in range(length):
        # Favour stepwise motion (±1-2 semitones in scale)
        idx = scale_notes.index(prev) if prev in scale_notes else 3
        step = random.choice([-2,-1,-1,0,1,1,2])
        new_idx = max(0, min(len(scale_notes)-1, idx + step))
        prev = scale_notes[new_idx]
        dur  = random.choice(DURATIONS)
        melody.append((prev, dur))
    return melody


def save_midi(melody, filename, tempo=120):
    """Save a melody list [(pitch, duration), ...] as a MIDI file."""
    midi = MIDIFile(1)
    midi.addTempo(0, 0, tempo)
    time = 0.0
    for pitch, dur in melody:
        midi.addNote(0, 0, pitch, time, dur, volume=80)
        time += dur
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        midi.writeFile(f)
    return path


def generate_dataset(n=20):
    """Generate n MIDI files across different scales."""
    scale_names = list(SCALES.keys())
    paths = []
    for i in range(n):
        scale_name = scale_names[i % len(scale_names)]
        scale      = SCALES[scale_name]
        tempo      = random.randint(80, 140)
        melody     = generate_melody(scale, length=48)
        fname      = f"piece_{i+1:02d}_{scale_name}_t{tempo}.mid"
        path       = save_midi(melody, fname, tempo=tempo)
        paths.append(path)
        print(f"  ✔ Generated {fname}")
    return paths


# ── Preprocessing with music21 ────────────────────────────────────────────────
def extract_notes(midi_path):
    """Parse a MIDI file and return a flat list of note/chord strings."""
    notes_out = []
    try:
        midi = converter.parse(midi_path)
        parts = instrument.partitionByInstrument(midi)
        elements = parts.parts[0] if parts else midi.flat
        for elem in elements.notes:
            if isinstance(elem, note.Note):
                notes_out.append(str(elem.pitch))
            elif isinstance(elem, chord.Chord):
                notes_out.append(".".join(str(n) for n in elem.normalOrder))
    except Exception as e:
        print(f"  ⚠ Could not parse {midi_path}: {e}")
    return notes_out


def preprocess_dataset(midi_paths):
    """Extract and flatten note sequences from all MIDI files."""
    all_notes = []
    for path in midi_paths:
        seq = extract_notes(path)
        all_notes.extend(seq)
        print(f"  ✔ Parsed {os.path.basename(path)} → {len(seq)} notes")
    return all_notes


if __name__ == "__main__":
    print("\n═══════════════════════════════════════════════")
    print("  STEP 1 – Generating MIDI training dataset")
    print("═══════════════════════════════════════════════")
    midi_paths = generate_dataset(n=20)

    print(f"\n  Total files generated: {len(midi_paths)}")

    print("\n═══════════════════════════════════════════════")
    print("  STEP 2 – Preprocessing with music21")
    print("═══════════════════════════════════════════════")
    all_notes = preprocess_dataset(midi_paths)

    np.save("all_notes.npy", all_notes)
    print(f"\n  Total note tokens extracted: {len(all_notes)}")
    print("  Saved  → all_notes.npy")
    print("\n  Unique tokens sample:", list(set(all_notes))[:10])
