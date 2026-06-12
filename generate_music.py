"""
Step 5: Generate New Music and Convert to MIDI
- Loads trained LSTM model
- Generates novel note sequences via temperature-based sampling
- Converts generated sequences → MIDI file
- (Optional) Converts MIDI → WAV using fluidsynth if available
"""

import os
import sys
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from midiutil import MIDIFile
from music21 import note, chord, stream, tempo as m21tempo, instrument

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN    = 16
GENERATE_N = 200     # notes to generate
TEMPERATURE = 1.0    # creativity: 0.5 = conservative, 1.2 = creative
TEMPO_BPM   = 120
OUTPUT_DIR  = "output"
MODELS_DIR  = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model_and_vocab():
    from tensorflow.keras.models import load_model
    model    = load_model(os.path.join(MODELS_DIR, "music_lstm_model.keras"))
    note2idx = np.load("note2idx.npy", allow_pickle=True).item()
    idx2note = np.load("idx2note.npy", allow_pickle=True).item()
    encoded  = np.load("encoded_notes.npy", allow_pickle=True).tolist()
    return model, note2idx, idx2note, encoded


def sample_with_temperature(predictions, temperature=TEMPERATURE):
    """Sample from softmax output with temperature scaling."""
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions + 1e-10) / temperature
    exp_preds   = np.exp(predictions)
    predictions = exp_preds / exp_preds.sum()
    return np.random.choice(len(predictions), p=predictions)


def generate_sequence(model, encoded, note2idx, idx2note,
                      n=GENERATE_N, temperature=TEMPERATURE, seq_len=SEQ_LEN):
    """Generate n new note tokens auto-regressively."""
    # Pick a random seed window from the training data
    start   = random.randint(0, len(encoded) - seq_len - 1)
    pattern = list(encoded[start : start + seq_len])
    generated = []

    print(f"  Seed notes: {[idx2note[i] for i in pattern[:5]]} …")

    for _ in range(n):
        x    = np.reshape(pattern, (1, seq_len))
        pred = model.predict(x, verbose=0)[0]
        idx  = sample_with_temperature(pred, temperature)
        generated.append(idx2note[idx])
        pattern.append(idx)
        pattern = pattern[1:]          # slide window

    return generated


def notes_to_midi(notes_list, filename, bpm=TEMPO_BPM):
    """Convert a list of note/chord strings to a MIDI file using midiutil."""
    midi  = MIDIFile(1)
    midi.addTempo(0, 0, bpm)
    time  = 0.0
    dur   = 0.5         # default duration per note

    # Note-name → MIDI pitch mapping (basic)
    from music21 import pitch as m21pitch
    for token in notes_list:
        try:
            if "." in token:
                # chord: token looks like "0.4.7"
                semitones = [int(s) for s in token.split(".")]
                base = 60
                for st in semitones:
                    midi.addNote(0, 0, base + st, time, dur, volume=75)
            else:
                p = m21pitch.Pitch(token)
                midi.addNote(0, 0, p.midi, time, dur, volume=80)
        except Exception:
            pass        # skip unparseable tokens
        time += dur

    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        midi.writeFile(f)
    return path


def notes_to_music21_midi(notes_list, filename, bpm=TEMPO_BPM):
    """
    Alternative: use music21 stream for richer MIDI with proper note durations.
    Saves alongside the midiutil version.
    """
    from music21 import note as m21note, chord as m21chord, stream as m21stream
    from music21 import tempo as m21tempo

    s = m21stream.Stream()
    s.append(m21tempo.MetronomeMark(number=bpm))

    for token in notes_list:
        try:
            if "." in token:
                semitones  = [int(x) for x in token.split(".")]
                pitches    = [60 + st for st in semitones]
                c          = m21chord.Chord(pitches)
                c.duration.quarterLength = 0.5
                s.append(c)
            else:
                n = m21note.Note(token)
                n.duration.quarterLength = 0.5
                s.append(n)
        except Exception:
            pass

    path = os.path.join(OUTPUT_DIR, filename)
    s.write("midi", fp=path)
    return path


if __name__ == "__main__":
    print("\n═══════════════════════════════════════════════")
    print("  STEP 5 – Generating New Music")
    print("═══════════════════════════════════════════════")

    model, note2idx, idx2note, encoded = load_model_and_vocab()
    print(f"  Model loaded   ✔")
    print(f"  Vocab size     : {len(note2idx)}")

    generated = generate_sequence(model, encoded, note2idx, idx2note,
                                   n=GENERATE_N, temperature=TEMPERATURE)
    print(f"\n  Generated {len(generated)} note tokens")
    print(f"  Sample: {generated[:10]}")

    # Save via midiutil
    out1 = notes_to_midi(generated, "generated_music_midiutil.mid")
    print(f"\n  MIDI saved (midiutil) → {out1}")

    # Save via music21
    out2 = notes_to_music21_midi(generated, "generated_music_music21.mid")
    print(f"  MIDI saved (music21)  → {out2}")

    # Try WAV conversion via fluidsynth (optional)
    import subprocess
    sf2_candidates = [
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        "/usr/share/soundfonts/FluidR3_GM.sf2",
    ]
    sf2 = next((p for p in sf2_candidates if os.path.exists(p)), None)
    if sf2:
        wav_out = os.path.join(OUTPUT_DIR, "generated_music.wav")
        result  = subprocess.run(
            ["fluidsynth", "-ni", sf2, out2, "-F", wav_out, "-r", "44100"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  WAV  saved        → {wav_out}")
        else:
            print(f"  WAV conversion skipped (fluidsynth error)")
    else:
        print("  WAV conversion skipped (no soundfont found; MIDI files are ready)")

    print("\n  ✅ Generation complete!")
