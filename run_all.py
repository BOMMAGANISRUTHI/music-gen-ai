"""
Music Generation with AI — Main Runner
Runs all 5 steps end-to-end:
  1. Generate MIDI training data
  2. Preprocess with music21
  3. Build LSTM model
  4. Train the model
  5. Generate new MIDI music
"""

import subprocess
import sys
import os
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

STEPS = [
    ("generate_midi_data.py", "Generate MIDI data + Preprocess with music21"),
    ("train_model.py",        "Build LSTM model + Train"),
    ("generate_music.py",     "Generate new music + Save MIDI"),
]

def run_step(script, label):
    print(f"\n{'═'*55}")
    print(f"  ▶  {label}")
    print(f"{'═'*55}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ✗ Step failed after {elapsed:.1f}s")
        sys.exit(1)
    print(f"\n  ✔ Done in {elapsed:.1f}s")

if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  🎵  Music Generation with AI  —  CodeAlpha Task 3")
    print("═"*55)
    total = time.time()
    for script, label in STEPS:
        run_step(script, label)
    print(f"\n{'═'*55}")
    print(f"  🎉  All steps complete! ({time.time()-total:.1f}s total)")
    print(f"  Output MIDI files are in: output/")
    print("═"*55)
