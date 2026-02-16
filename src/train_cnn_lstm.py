import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping

from data_loader import load_ravdess_dataset


# =============================
# 1. LOAD DATASET
# =============================

X, y = load_ravdess_dataset("data/RAVDESS")

print("Loaded X shape:", X.shape)

if X.size == 0:
    raise ValueError("Dataset is empty. Check working directory.")

# =============================
# 2. FIX SHAPE SAFELY
# =============================

# If 4D (from CNN previous version)
if len(X.shape) == 4:
    X = X.squeeze(-1)
    print("After squeeze:", X.shape)

# Ensure correct shape
if len(X.shape) != 3:
    raise ValueError(f"Unexpected X shape: {X.shape}")

# Transpose for LSTM: (samples, 40, 174) → (samples, 174, 40)
X = np.transpose(X, (0, 2, 1))
print("After transpose:", X.shape)

# =============================
# 3. ENCODE LABELS
# =============================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# =============================
# 4. TRAIN-TEST SPLIT
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =============================
# 5. BUILD CNN + LSTM MODEL
# =============================

model = Sequential([

    # CNN Block 1
    Conv1D(64, kernel_size=3, activation="relu",
           input_shape=X_train.shape[1:]),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # CNN Block 2
    Conv1D(128, kernel_size=3, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # LSTM
    LSTM(128, return_sequences=False),
    Dropout(0.4),

    # Dense classifier
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(len(label_encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =============================
# 6. TRAIN MODEL
# =============================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# =============================
# 7. EVALUATE
# =============================

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", test_accuracy)

y_pred = model.predict(X_test).argmax(axis=1)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - CNN + LSTM")
plt.tight_layout()
plt.show()
# =============================
#  Plot Training Curves
# =============================

plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

import os
import pickle

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Save model
model.save("models/cnn_lstm_model.h5")
print("Model saved successfully.")

# Save label encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Label encoder saved successfully.")



