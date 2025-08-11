# app.py — FINAL
# Streamlit Cloud-friendly, low-memory, lazy TF import, optional GDrive download.

import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -------- SETTINGS (edit only these if needed) --------
GDRIVE_MODEL_ID  = None   # e.g. "1AbC..."; keep None if file is in repo
GDRIVE_SCALER_ID = None
GDRIVE_LABELS_ID = None

MODEL_PATH  = Path("motion_model_v2.h5")   # your Keras model saved with include_optimizer=False (smaller)
SCALER_PATH = Path("scaler.pkl")
LABELS_PATH = Path("label_map.json")

DEFAULT_WINDOW_SIZE = 50     # same window length used in training
MAX_ROWS = 10000             # hard cap to avoid OOM on Streamlit Cloud
BATCH_WINDOWS = 256          # process this many windows per batch
# ------------------------------------------------------

# keep TF light + fewer threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

def ensure_file(local: Path, file_id: str | None):
    if local.exists() or not file_id:
        return
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    st.info(f"Downloading {local.name}…")
    gdown.download(url, str(local), quiet=False)

@st.cache_resource(show_spinner=True)
def load_assets():
    # download big files if needed
    ensure_file(MODEL_PATH,  GDRIVE_MODEL_ID)
    ensure_file(SCALER_PATH, GDRIVE_SCALER_ID)
    ensure_file(LABELS_PATH, GDRIVE_LABELS_ID)

    # lazy import TF here
    from tensorflow.keras.models import load_model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH.name}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing {SCALER_PATH.name}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing {LABELS_PATH.name}")

    model = load_model(str(MODEL_PATH))
    scaler = joblib.load(str(SCALER_PATH))
    with open(LABELS_PATH) as f:
        raw = json.load(f)
        label_map = {int(k): v for k, v in raw.items()}
    return model, scaler, label_map

def window_pairs(T, window, step):
    i = 0
    while i + window <= T:
        yield i, i + window
        i += step

def predict_streaming(model, scaler, df, window_size, label_map, step=None, batch=BATCH_WINDOWS):
    if step is None:
        step = window_size
    X = df.select_dtypes(include=[np.number]).to_numpy()
    if X.ndim != 2:
        raise ValueError("CSV must be numeric columns only (same features as training).")
    T, F = X.shape
    if T < window_size:
        raise ValueError(f"Need at least {window_size} rows; got {T}.")
    preds_idx, buf = [], []

    for a, b in window_pairs(T, window_size, step):
        buf.append(X[a:b])
        if len(buf) == batch:
            W = np.stack(buf)                       # (B, w, F)
            n, w, f = W.shape
            Ws = scaler.transform(W.reshape(-1, f)).reshape(n, w, f)
            p = model.predict(Ws, verbose=0)
            preds_idx.extend(np.argmax(p, axis=1).tolist())
            buf.clear()

    if buf:
        W = np.stack(buf)
        n, w, f = W.shape
        Ws = scaler.transform(W.reshape(-1, f)).reshape(n, w, f)
        p = model.predict(Ws, verbose=0)
        preds_idx.extend(np.argmax(p, axis=1).tolist())

    labels = [label_map[int(i)] for i in preds_idx]
    out = pd.DataFrame({"Window": np.arange(len(labels)) + 1,
                        "Predicted Activity": labels})
    return out, labels

# ---------------- UI ----------------
st.set_page_config(page_title="MotionSense", page_icon="🏃", layout="centered")
st.title("🏃 MotionSense Activity Classifier")

with st.sidebar:
    st.subheader("Settings")
    window_size = st.number_input("Window size", 5, 512, DEFAULT_WINDOW_SIZE, 5)
    st.caption("Files expected in repo root or auto-downloaded from Drive:")
    st.code(f"{MODEL_PATH.name}\n{SCALER_PATH.name}\n{LABELS_PATH.name}")

file = st.file_uploader("Upload sensor CSV", type=["csv"])
if not file:
    st.info("Upload a CSV with the same numeric features used in training.")
    st.stop()

df = pd.read_csv(file)
if len(df) > MAX_ROWS:
    st.warning(f"Trimming rows: {len(df):,} → {MAX_ROWS:,} to fit free tier memory.")
    df = df.iloc[:MAX_ROWS]

st.subheader("Preview")
st.dataframe(df.head(10), use_container_width=True)

try:
    with st.spinner("Loading model…"):
        model, scaler, label_map = load_assets()
    with st.spinner("Predicting…"):
        pred_df, labels = predict_streaming(model, scaler, df, window_size, label_map)
    st.subheader("Results (per window)")
    st.dataframe(pred_df, use_container_width=True)

    st.subheader("Summary")
    st.write(pd.Series(labels).value_counts().to_frame("Count"))

    st.subheader("Recommendations")
    tips = {
        "sitting": "Stand and stretch every 30 min.",
        "standing": "Relax shoulders; shift weight.",
        "walking": "Great—aim for 7–10k steps/day.",
        "jogging": "Hydrate and warm up/cool down.",
        "upstairs": "Use rail if needed.",
        "downstairs": "Slow, controlled steps."
    }
    for act in pred_df["Predicted Activity"].unique()[:4]:
        st.markdown(f"**{act}** → {tips.get(act.lower(), 'Keep moving!')}")

except Exception as e:
    st.error(f"❌ {e}")
