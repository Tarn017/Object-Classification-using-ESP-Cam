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

received_values = []  # Liste zum Speichern der empfangenen Werte

################################################################################

# ESP32-CAM Endpunkt
esp32_url = 'http://192.168.1.100/capture'

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

#########################################################################################################

try:
    while True:
        # Daten vom Arduino empfangen
        response = client_socket.recv(1024).decode().strip()

        if response:  # Falls eine Antwort empfangen wurde
            try:
                value = int(response)  # Versuche, die Antwort in eine Zahl umzuwandeln
                received_values.append(value)  # Wert in die Liste speichern
                print(f"Empfangen und gespeichert: {value}")
                if(value == 42):
                    image_path = capture_image()
                    if image_path:
                        result, confidence = classify_image(image_path)
                        print(f"Das Bild enthält: {result} (Sicherheit: {confidence:.2f})")
                        client_socket.sendall((result + '\n').encode())

            except ValueError:
                print(f"Fehler beim Konvertieren: {response}")  # Falls ungültige Daten empfangen werden

        # Falls du eine Bedingung zum Stoppen willst (z. B. nach 10 Werten)
        if len(received_values) >= 10:
            print("Genug Werte empfangen, beende die Verbindung.")
            break

finally:
    client_socket.close()

# Nach dem Programmende: Gespeicherte Werte anzeigen
print(f"Gespeicherte Werte: {received_values}")
