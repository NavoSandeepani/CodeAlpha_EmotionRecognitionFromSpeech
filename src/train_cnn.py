import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential


from data_loader import load_ravdess_dataset

#load dataset
x,y=load_ravdess_dataset("data/RAVDESS")
#encode label(text -> number)
label_encoder=LabelEncoder()
y_encoded=label_encoder.fit_transform(y)

#add channel diamensional
x=x[...,np.newaxis]

#train_test
x_train,x_test,y_train,y_test=train_test_split(x,y_encoded,test_size=0.2,random_state=42,stratify=y_encoded)

#build cnn model
model=Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=x_train.shape[1:]),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation="softmax") 


])
# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_accuracy)

# Predictions
y_pred = model.predict(x_test).argmax(axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))