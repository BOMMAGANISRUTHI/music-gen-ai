# 🎵 Music Generation with AI — CodeAlpha Task 3

An AI-powered music composer that learns from MIDI data and generates original melodies using a deep learning LSTM model.

---

## 📌 Project Overview

This project trains a Long Short-Term Memory (LSTM) neural network on synthetic classical-style MIDI sequences and uses it to compose new music. It demonstrates the full pipeline from data generation to AI-composed MIDI output.

---

## 🚀 How It Works

**Step 1 — Generate Training Data**
Generates 20 synthetic MIDI files across 5 musical scales (C Major, G Major, A Minor, D Major, F Major).

**Step 2 — Preprocess with music21**
Parses MIDI files and extracts note sequences (1037 note tokens total).

**Step 3 — Build LSTM Model**
Constructs a 2-layer LSTM neural network:
`Embedding → LSTM(256) → Dropout → LSTM(128) → Dropout → Dense(softmax)`

**Step 4 — Train the Model**
Trains for 30 epochs using Adam optimizer, achieving ~51% accuracy.

**Step 5 — Generate New Music**
Uses temperature-based sampling to compose original melodies and saves them as MIDI files.

---

## 🛠️ Tech Stack

- **Python 3.10**
- **TensorFlow / Keras** — LSTM model
- **music21** — MIDI parsing and music theory
- **midiutil** — MIDI file generation
- **NumPy** — Data processing

---

## 📁 Project Structure

```
music_gen_ai/
├── generate_midi_data.py   # Step 1 & 2: Generate + preprocess MIDI data
├── train_model.py          # Step 3 & 4: Build + train LSTM model
├── generate_music.py       # Step 5: Generate new music
├── run_all.py              # Run all steps end-to-end
├── midi_data/              # Generated training MIDI files
├── models/                 # Saved LSTM model
└── output/                 # AI-generated music output
```

---

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/BOMMAGANISRUTHI/music-gen-ai.git
cd music-gen-ai
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install midiutil music21 tensorflow numpy
```

**4. Run the full pipeline**
```bash
python run_all.py
```

**5. Listen to your AI-generated music**

Find the output MIDI files in the `output/` folder and open with any MIDI player or at [https://signal.vercel.app](https://signal.vercel.app).

---

## 🎼 Output

The model generates two MIDI files:
- `generated_music_midiutil.mid` — generated via midiutil
- `generated_music_music21.mid` — generated via music21 (richer output)

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Epochs | 30 |
| Final Loss | 1.2156 |
| Final Accuracy | 51.22% |
| Vocabulary Size | 16 notes |
| Training Samples | 1021 |

---

## 👩‍💻 Author

**Sruthi Bommagani**
CodeAlpha Internship — Task 3: Music Generation with AI
