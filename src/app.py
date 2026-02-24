import streamlit as st
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/cnn_lstm_model.h5")

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

N_MFCC = 40
MAX_LEN = 174

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc


st.title("🎤 Emotion Recognition from Speech")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    mfcc = extract_mfcc("temp.wav")
    mfcc = np.transpose(mfcc, (1, 0))
    mfcc = np.expand_dims(mfcc, axis=0)

    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    st.success(f"Predicted Emotion: {predicted_label}")