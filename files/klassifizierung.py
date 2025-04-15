import requests
import tensorflow as tf
import numpy as np
from tensorflow import keras

# ESP32-CAM Endpunkt
esp32_url = 'http://192.168.1.100/capture'

# Bildparameter
img_height = 240
img_width = 320

# Klassenbezeichnungen
class_names = ['Noah', 'Tabea', 'Tobias','noise']

# Modell laden
model = keras.models.load_model("objekt_klassifikator.keras")

def capture_image():
    """Nimmt ein einzelnes Bild von der ESP32-CAM auf und speichert es unter dem gleichen Namen."""
    filename = 'latest_capture.jpg'
    try:
        response = requests.get(esp32_url, timeout=5)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f'Bild gespeichert: {filename}')
            return filename
        else:
            print(f'Fehler: Statuscode {response.status_code}')
    except requests.RequestException as e:
        print(f'Anfrage fehlgeschlagen: {e}')
    return None

def classify_image(image_path):
    """Klassifiziert ein einzelnes Bild mit dem geladenen Modell."""
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])  # Index der höchsten Wahrscheinlichkeit
    confidence = np.max(predictions[0])  # Höchste Wahrscheinlichkeit
    print(f"Rohwerte der Vorhersage: {predictions[0]}")
    return class_names[predicted_class], confidence

# Bild aufnehmen und klassifizieren
image_path = capture_image()
if image_path:
    result, confidence = classify_image(image_path)
    print(f"Das Bild enthält: {result} (Sicherheit: {confidence:.2f})")
