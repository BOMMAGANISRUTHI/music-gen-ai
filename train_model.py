"""
Step 3 & 4: Build a Deep Learning LSTM Model and Train It
- Encodes note sequences
- Builds an LSTM-based neural network
- Trains on the sequences
- Saves model weights
"""

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# ── Hyper-parameters ──────────────────────────────────────────────────────────
SEQ_LEN   = 16      # notes fed as context
EPOCHS    = 30
BATCH     = 64
EMBED_DIM = 64
LSTM_UNITS= 256

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_encode(notes_path="all_notes.npy"):
    """Load note tokens, build vocab, return integer sequences."""
    notes      = np.load(notes_path, allow_pickle=True).tolist()
    vocab      = sorted(set(notes))
    note2idx   = {n: i for i, n in enumerate(vocab)}
    idx2note   = {i: n for n, i in note2idx.items()}
    encoded    = [note2idx[n] for n in notes]
    print(f"  Vocabulary size : {len(vocab)}")
    print(f"  Total tokens    : {len(encoded)}")
    return encoded, note2idx, idx2note


def build_sequences(encoded, seq_len=SEQ_LEN):
    """Slide a window over the encoded sequence to create (X, y) pairs."""
    X, y = [], []
    for i in range(len(encoded) - seq_len):
        X.append(encoded[i : i + seq_len])
        y.append(encoded[i + seq_len])
    X = np.array(X)
    y = np.array(y)
    print(f"  Training samples: {len(X)}")
    return X, y


def build_lstm_model(vocab_size, seq_len=SEQ_LEN, embed_dim=EMBED_DIM, lstm_units=LSTM_UNITS):
    """
    LSTM model:
      Embedding → LSTM → Dropout → LSTM → Dropout → Dense(softmax)
    """
    model = Sequential([
        Embedding(vocab_size, embed_dim, input_length=seq_len),
        LSTM(lstm_units, return_sequences=True),
        Dropout(0.3),
        LSTM(lstm_units // 2),
        Dropout(0.3),
        Dense(vocab_size, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model


def train(model, X, y, epochs=EPOCHS, batch_size=BATCH):
    """Train with checkpointing and early stopping."""
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, "best_model.weights.h5"),
        monitor="loss", save_best_only=True, save_weights_only=True, verbose=0
    )
    early_stop = EarlyStopping(monitor="loss", patience=5, verbose=1)
    reduce_lr  = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, verbose=1)

    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    return history


if __name__ == "__main__":
    print("\n═══════════════════════════════════════════════")
    print("  STEP 3 – Building LSTM Model")
    print("═══════════════════════════════════════════════")

    encoded, note2idx, idx2note = load_and_encode()
    vocab_size = len(note2idx)

    np.save("note2idx.npy", note2idx)
    np.save("idx2note.npy", idx2note)
    np.save("encoded_notes.npy", encoded)

    X, y = build_sequences(encoded)

    model = build_lstm_model(vocab_size)
    model.summary()

    print("\n═══════════════════════════════════════════════")
    print("  STEP 4 – Training the Model")
    print("═══════════════════════════════════════════════")
    history = train(model, X, y)

    # Save full model
    model.save(os.path.join(MODELS_DIR, "music_lstm_model.keras"))
    print("\n  Model saved → models/music_lstm_model.keras")

    final_loss = history.history["loss"][-1]
    final_acc  = history.history["accuracy"][-1]
    print(f"  Final loss     : {final_loss:.4f}")
    print(f"  Final accuracy : {final_acc:.4f}")
