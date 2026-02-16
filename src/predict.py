import sys
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

# -----------------------------
# SETTINGS
# -----------------------------
N_MFCC = 40
MAX_LEN = 174

# -----------------------------
# LOAD MODEL & LABEL ENCODER
# -----------------------------
model = load_model("models/cnn_lstm_model.h5")

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# -----------------------------
# FEATURE EXTRACTION FUNCTION
# -----------------------------
def extract_mfcc(file_path, n_mfcc=N_MFCC, max_len=MAX_LEN):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Padding or truncating
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc


# -----------------------------
# MAIN PREDICTION
# -----------------------------
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path_to_audio.wav")
        sys.exit(1)

    file_path = sys.argv[1]

    print(f"Processing file: {file_path}")

    # Extract MFCC
    mfcc = extract_mfcc(file_path)

    # Shape correction for CNN+LSTM
    mfcc = np.transpose(mfcc, (1, 0))  # (174, 40)
    mfcc = np.expand_dims(mfcc, axis=0)  # (1, 174, 40)

    # Predict
    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    print("Predicted Emotion:", predicted_label)
