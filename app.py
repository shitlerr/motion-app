
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model
import os
from pathlib import Path
import gdown
import joblib, json, pandas as pd, numpy as np, streamlit as st

MODEL_PATH = Path("motion_model_v2.h5")
SCALER_PATH = Path("scaler.pkl")
LABELS_PATH = Path("label_map.json")

def ensure_file(local_path: Path, gdrive_id: str):
    if not local_path.exists():
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, str(local_path), quiet=False)

# TODO: replace with your real file IDs
ensure_file(MODEL_PATH,  "YOUR_MODEL_FILE_ID")
ensure_file(SCALER_PATH, "YOUR_SCALER_FILE_ID")
ensure_file(LABELS_PATH, "YOUR_LABELS_FILE_ID")

model  = load_model(str(MODEL_PATH))
scaler = joblib.load(str(SCALER_PATH))
with open(LABELS_PATH, "r") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}


# Load model and helper files
model = load_model("motion_model_v2.h5")
scaler = joblib.load("scaler.pkl")

with open("label_map.json", "r") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}

st.title("🏃‍♀️ MotionSense Activity Classifier")
st.write("Upload your motion sensor data to classify physical activity and receive personalized advice.")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.write(df.head())

    try:
        x = df.values
        x_scaled = scaler.transform(x)
        preds = model.predict(x_scaled)
        y_pred = np.argmax(preds, axis=1)
        df["Predicted Activity"] = [label_map[i] for i in y_pred]

        st.subheader("✅ Prediction Results")
        st.write(df[["Predicted Activity"]])

        st.subheader("💡 Recommendations")
        for activity in df["Predicted Activity"].unique():
            st.write(f"**{activity}** → ", end="")
            if activity.lower() == "sitting":
                st.write("Get up and stretch every 30 minutes!")
            elif activity.lower() == "walking":
                st.write("Nice! Keep walking for a healthy heart.")
            elif activity.lower() == "jogging":
                st.write("Good job! Stay hydrated.")
            else:
                st.write("Keep it up!")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
