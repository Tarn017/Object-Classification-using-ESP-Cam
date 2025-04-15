import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models

# Datenverzeichnis anpassen
train_dir = "daten/"

# Bildparameter
img_height = 240  # Höhe des Bildes
img_width = 320   # Breite des Bildes
batch_size = 32

# Trainings- und Validierungsdatensätze erstellen
dataset = image_dataset_from_directory(
    train_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'  # Mehrklassen-Klassifikation
)

# Dataset normalisieren
normalization_layer = layers.Rescaling(1./255)
dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

# CNN-Modell definieren
model = models.Sequential()

# Block 1
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(240, 320, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Block 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Block 3
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Block 4
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flach- und Vollverbundene Schichten
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))  # 4 statt 3 Klassen

# Modellkompilierung
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modell trainieren
history = model.fit(dataset, epochs=15)  # Weniger Epochen zum Verhindern von Overfitting

# Modell speichern
model.save("objekt_klassifikator.keras")
