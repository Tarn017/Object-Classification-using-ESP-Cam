import socket
import requests
import tensorflow as tf
import numpy as np
from tensorflow import keras

arduino_ip = '192.168.1.102'  # IP-Adresse des Arduino
port = 12345  # Muss mit der Portnummer im Arduino-Sketch übereinstimmen

# Socket erstellen
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((arduino_ip, port))

################################################################################

# ESP32-CAM Endpunkt
esp32_url = 'http://192.168.1.101/capture'

# Bildparameter
img_height = 240
img_width = 320

# Klassenbezeichnungen
class_names = ['noise', 'schachtel', 'tesa']

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

###################################################################################################

try:
    while True:
        # Zahl vom Benutzer eingeben
        number = input("Geben Sie eine Zahl ein (oder 'exit' zum Beenden): ")
        if number.lower() == 'exit':
            break
        # Zahl an den Arduino senden
        client_socket.sendall((number + '\n').encode())
finally:
    client_socket.close()
