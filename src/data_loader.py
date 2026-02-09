import os
import numpy as np
from feature_extraction import extract_mfcc

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

def get_label_from_filename(filename: str) -> str:
    parts = filename.split("-")
    if len(parts) < 3:
        return "unknown"
    emotion_code = parts[2]
    return EMOTION_MAP.get(emotion_code, "unknown")

def load_ravdess_dataset(data_path="data/RAVDESS", n_mfcc=40, max_len=174):
    X, y = [], []

    print("Walking path:", data_path)

    for root, _, files in os.walk(data_path):
        print("Entered folder:", root)
        for f in files:
            print("Found file:", f)

            if f.lower().endswith(".wav"):
                label = get_label_from_filename(f)
                print("Detected emotion:", label)

                if label == "unknown":
                    continue

                file_path = os.path.join(root, f)
                mfcc = extract_mfcc(file_path, n_mfcc=n_mfcc, max_len=max_len)

                X.append(mfcc)
                y.append(label)

    return np.array(X), np.array(y)
