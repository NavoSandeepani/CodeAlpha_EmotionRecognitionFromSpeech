import streamlit as st
import numpy as np
import librosa
import pickle
import os
from tensorflow.keras.models import load_model

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤 Emotion Recognition from Speech")
st.markdown("Upload or record your voice to detect emotion instantly!")

# -----------------------------
# LOAD MODEL & LABEL ENCODER
# -----------------------------
@st.cache_resource
def load_resources():
    model = load_model("models/cnn_lstm_model.h5")
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_resources()

# -----------------------------
# SETTINGS
# -----------------------------
N_MFCC = 40
MAX_LEN = 174

# -----------------------------
# EMOJI MAP
# -----------------------------
emoji_map = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😡",
    "calm": "😌",
    "neutral": "😐",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😲"
}

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc


def predict_emotion(file_path):
    mfcc = extract_mfcc(file_path)
    mfcc = np.transpose(mfcc, (1, 0))
    mfcc = np.expand_dims(mfcc, axis=0)

    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(np.max(prediction)) * 100

    return predicted_label, confidence


# -----------------------------
# FILE UPLOAD SECTION
# -----------------------------
st.markdown("### 📂 Upload WAV File")
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    label, confidence = predict_emotion("temp.wav")
    emoji = emoji_map.get(label, "")

    st.success(f"## {emoji} {label.upper()}")
    st.info(f"Confidence: {confidence:.2f}%")

# -----------------------------
# MICROPHONE RECORDING SECTION
# -----------------------------
st.markdown("---")
st.markdown("### 🎙️ Record Your Voice")

audio_bytes = st.audio_input("Click to record")

if audio_bytes is not None:
    st.audio(audio_bytes)

    with open("recorded.wav", "wb") as f:
        f.write(audio_bytes.getbuffer())

    label, confidence = predict_emotion("recorded.wav")
    emoji = emoji_map.get(label, "")

    st.success(f"## {emoji} {label.upper()}")
    st.info(f"Confidence: {confidence:.2f}%")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Deep Learning Model: CNN + LSTM | Accuracy: ~79%")