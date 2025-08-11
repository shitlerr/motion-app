# app.py
# Streamlit app for MotionSense activity classification
# - Works on Streamlit Cloud
# - Lazy-loads TensorFlow
# - Can pull large files from Google Drive (set the IDs below)

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ---------- CONFIG ----------
# If your files are already in the repo root, leave these IDs as None.
# If your model/scaler/labels are too large for GitHub, upload them to Google Drive,
# set sharing to "Anyone with the link", then put the FILE IDs here.
GDRIVE_MODEL_ID = None          # e.g. "1AbCDeFG..."; or None
GDRIVE_SCALER_ID = None         # or None
GDRIVE_LABELS_ID = None         # or None

# If your DL model was trained on windows (CNN/LSTM), set your window size here.
DEFAULT_WINDOW_SIZE = 50

# Expected file names in the app working dir:
MODEL_PATH  = Path("motion_model_v2.h5")   # Keras .h5
SCALER_PATH = Path("scaler.pkl")
LABELS_PATH = Path("label_map.json")
# ----------------------------


def ensure_file(local_path: Path, file_id: str | None):
    """Download from Google Drive if file_id is provided and file is missing."""
    if local_path.exists() or not file_id:
        return
    import gdown  # lightweight; only imported when needed
    url = f"https://drive.google.com/uc?id={file_id}"
    st.info(f"Downloading {local_path.name} from Google Drive…")
    gdown.download(url, str(local_path), quiet=False)


@st.cache_resource(show_spinner=True)
def load_assets():
    """Load model, scaler, and label map. TensorFlow is imported lazily here."""
    # Download large files if IDs were provided
    ensure_file(MODEL_PATH,  GDRIVE_MODEL_ID)
    ensure_file(SCALER_PATH, GDRIVE_SCALER_ID)
    ensure_file(LABELS_PATH, GDRIVE_LABELS_ID)

    # Lazy TF import so the app boots fast
    from tensorflow.keras.models import load_model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH.name}. Add it to the repo root or set GDRIVE_MODEL_ID."
        )
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SCALER_PATH.name}. Add it to the repo root or set GDRIVE_SCALER_ID."
        )
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {LABELS_PATH.name}. Add it to the repo root or set GDRIVE_LABELS_ID."
        )

    model = load_model(str(MODEL_PATH))
    scaler = joblib.load(str(SCALER_PATH))
    with open(LABELS_PATH, "r") as f:
        raw_map = json.load(f)
        # keys might be strings; normalize to int
        label_map = {int(k): v for k, v in raw_map.items()}

    return model, scaler, label_map


def make_windows(X: np.ndarray, window_size: int, step: int | None = None) -> np.ndarray:
    """Slice (T, F) array into (N, window_size, F) windows."""
    if step is None:
        step = window_size
    T = X.shape[0]
    if T < window_size:
        return np.empty((0, window_size, X.shape[1]))
    idxs = range(0, T - window_size + 1, step)
    windows = np.stack([X[i:i+window_size] for i in idxs], axis=0)
    return windows


def predict_windows(model, scaler, df: pd.DataFrame, window_size: int, label_map: dict[int, str]):
    """Predict per-window labels for CNN/LSTM models trained on sequence windows."""
    # Use only numeric columns in the same order as the CSV
    X = df.select_dtypes(include=[np.number]).to_numpy()
    if X.ndim != 2:
        raise ValueError("Uploaded CSV must be a 2D table of numeric sensor features.")

    # Form windows
    W = make_windows(X, window_size=window_size, step=window_size)
    if W.size == 0:
        raise ValueError(f"Not enough rows ({len(X)}) for window_size={window_size}.")

    # Scale per-frame features with the same scaler used during training
    n_w, w, f = W.shape
    W_scaled = scaler.transform(W.reshape(-1, f)).reshape(n_w, w, f)

    # Predict
    probs = model.predict(W_scaled, verbose=0)
    y_idx = np.argmax(probs, axis=1)
    y_lbl = [label_map[int(i)] for i in y_idx]

    # Build a small summary dataframe
    out = pd.DataFrame({
        "Window": np.arange(len(y_lbl)) + 1,
        "Predicted Activity": y_lbl
    })
    return out, y_lbl


# ---------------- UI ----------------
st.set_page_config(page_title="MotionSense Classifier", page_icon="🏃", layout="centered")
st.title("🏃 MotionSense Activity Classifier")
st.caption("Upload your motion sensor CSV to classify activities. Works with windowed CNN/LSTM models.")

with st.sidebar:
    st.subheader("Settings")
    window_size = st.number_input("Window size (timesteps)", min_value=5, max_value=512,
                                  value=DEFAULT_WINDOW_SIZE, step=5)
    st.markdown("Files expected in the app root:")
    st.code("\n".join([
        MODEL_PATH.name,
        SCALER_PATH.name,
        LABELS_PATH.name
    ]), language="text")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV with the same numeric feature columns used in training.")
    st.stop()

# Show a quick preview
df_in = pd.read_csv(uploaded)
st.subheader("📊 Preview")
st.dataframe(df_in.head(10), use_container_width=True)

try:
    with st.spinner("Loading model and assets…"):
        model, scaler, label_map = load_assets()

    with st.spinner("Predicting…"):
        pred_df, labels = predict_windows(model, scaler, df_in, window_size, label_map)

    st.subheader("✅ Predictions (per window)")
    st.dataframe(pred_df, use_container_width=True)

    st.subheader("📈 Summary")
    counts = pd.Series(labels).value_counts()
    st.write(counts.to_frame("Count"))

    # Simple tips
    st.subheader("💡 Recommendations")
    tips = {
        "sitting": "Stand up and stretch every 30 minutes.",
        "standing": "Shift weight and relax shoulders to avoid stiffness.",
        "walking": "Great! Aim for 7–10k steps/day.",
        "jogging": "Nice pace — remember to hydrate.",
        "upstairs": "Use handrails if needed; watch your step.",
        "downstairs": "Slow down and plant your feet to reduce impact."
    }
    for act in counts.index.tolist()[:4]:
        msg = tips.get(act.lower(), "Keep moving and listen to your body.")
        st.markdown(f"**{act}** → {msg}")

except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()
