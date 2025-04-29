import requests
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
import numpy as np
import socket
from ultralytics import YOLO
from roboflow import Roboflow
from PIL import Image
import cv2

def abfrage(url, interval, klasse):
    # URL des ESP32-CAM /capture Endpunkts
    url = url

    # Intervall zwischen den Aufnahmen in Sekunden
    interval = interval

    # Pfad zum Speicherordner
    save_path = 'daten/'+klasse

    # Überprüfen, ob der Speicherordner existiert, andernfalls erstellen
    os.makedirs(save_path, exist_ok=True)

    while True:
        try:
            # Bild vom /capture Endpunkt abrufen
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                # Aktuellen Zeitstempel für den Dateinamen
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = os.path.join(save_path, f'capture_{timestamp}.jpg')

                # Bild speichern
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f'Bild gespeichert: {filename}')
            else:
                print(f'Fehler: Statuscode {response.status_code}')
        except requests.RequestException as e:
            print(f'Anfrage fehlgeschlagen: {e}')

        # Warten bis zum nächsten Abruf
        time.sleep(interval)

def get_class_names(train_dir):
    """
    Gibt eine alphabetisch sortierte Liste der Klassenbezeichnungen zurück,
    basierend auf den Unterordnern im angegebenen Trainingsverzeichnis.
    """
    return sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])

def training_classification(ordner, model_name, epochen):
    # Datenverzeichnis anpassen
    train_dir = ordner + '/'

    # Bildparameter
    img_height = 240  # Höhe des Bildes
    img_width = 320  # Breite des Bildes
    batch_size = 32

    # Trainings- und Validierungsdatensätze erstellen
    dataset = image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'  # Mehrklassen-Klassifikation
    )
    num_classes = len(dataset.class_names)

    # Dataset normalisieren
    normalization_layer = layers.Rescaling(1. / 255)
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
    model.add(layers.Dense(num_classes, activation='softmax'))  # 4 statt 3 Klassen

    # Modellkompilierung
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Modell trainieren
    history = model.fit(dataset, epochs=epochen)  # Weniger Epochen zum Verhindern von Overfitting

    # Modell speichern
    model.save(model_name)

def testen_classification(url, model, ordner):
    # Bildparameter
    img_height = 240
    img_width = 320

    # Klassenbezeichnungen
    class_names = get_class_names(ordner)
    print(class_names)

    # Modell laden
    model = keras.models.load_model("objekt_klassifikator.keras")

    def capture_image():
        """Nimmt ein einzelnes Bild von der ESP32-CAM auf und speichert es unter dem gleichen Namen."""
        filename = 'latest_capture.jpg'
        try:
            response = requests.get(url, timeout=5)
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
        return result, confidence
    return None

def neural_network_classification(url, arduino_ip, ordner, port):
    arduino_ip = arduino_ip  # IP-Adresse des Arduino
    port = port  # Muss mit der Portnummer im Arduino-Sketch übereinstimmen

    # Socket erstellen
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((arduino_ip, port))

    received_values = []  # Liste zum Speichern der empfangenen Werte

    ################################################################################

    # ESP32-CAM Endpunkt
    esp32_url = url

    # Bildparameter
    img_height = 240
    img_width = 320

    # Klassenbezeichnungen
    class_names = get_class_names(ordner)
    print(class_names)

    # Modell laden
    model = keras.models.load_model("objekt_klassifikator.keras")

    def capture_image():
        """Nimmt ein einzelnes Bild von der ESP32-CAM auf und speichert es unter dem gleichen Namen."""
        filename = 'latest_capture.jpg'
        try:
            response = requests.get(url, timeout=5)
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
                    if (value == 42):
                        image_path = capture_image()
                        if image_path:
                            result, confidence = classify_image(image_path)
                            result_str = f"{result},{confidence}\n"
                            print(f"Das Bild enthält: {result} (Sicherheit: {confidence:.2f})")
                            client_socket.sendall(result_str.encode())

                except ValueError:
                    print(f"Fehler beim Konvertieren: {response}")  # Falls ungültige Daten empfangen werden

            # Falls du eine Bedingung zum Stoppen willst (z. B. nach 10 Werten)
            if len(received_values) >= 10:
                print("Genug Werte empfangen, beende die Verbindung.")
                break
    finally:
        client_socket.close()

def training_detection(version, epochen):
    dataset = version.download("yolov8")

    # YOLO-Modell laden
    model = YOLO('yolov8n.pt')

    # Training starten
    results = model.train(data=os.path.join(dataset.location, 'data.yaml'), epochs=epochen, imgsz=320)

    # Modell auf TEST-Daten evaluieren
    metrics = model.val(data=os.path.join(dataset.location, 'data.yaml'), split='test')

    # Vorhersagen auf Testdaten visualisieren
    test_images_path = os.path.join(dataset.location, 'test', 'images')
    model.predict(source=test_images_path, conf=0.2, save=True)

def capture_image(url):
    """Nimmt ein einzelnes Bild von der ESP32-CAM auf und speichert es unter dem gleichen Namen."""
    filename = 'latest_capture.jpg'
    try:
        response = requests.get(url, timeout=5)
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

def testen_detection(url, model, conf_thresh):
    model = YOLO(model)
    capture_image(url)
    results = model.predict(source='latest_capture.jpg', conf=conf_thresh)
    # Ergebnisse extrahieren (Liste von Result-Objekten, hier nur 1 Bild → results[0])
    result = results[0]
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])  # Klassenindex (z. B. 0 für „ball“)
        label = model.names[cls_id]  # Klassennamen (z. B. "ball")
        conf = float(box.conf[0])  # Konfidenzscore
        xyxy = box.xyxy[0].tolist()  # Bounding Box Koordinaten [x1, y1, x2, y2]
        detections.append({
            "label": label,
            "confidence": conf,
            "bbox": xyxy
        })
    print(detections)

    annotated_img = result.plot()
    annotated_pil = Image.fromarray(annotated_img)
    annotated_pil.save('annotated_latest.jpg')

    # 3 Sekunden warten
    if detections:
        # Nur das erste erkannte Objekt verwenden
        det = detections[0]
        label = det['label']
        conf = round(det['confidence'], 2)
        x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
    else:
        # Kein Objekt erkannt
        label = "none"
        conf = 0
        x1 = y1 = x2 = y2 = 0

    # Arduino-freundlicher, einfacher CSV-String
    result_str = f"{label},{conf},{x1},{y1},{x2},{y2}\n"
    return result_str

def neural_network_detection(url, arduino_ip, port, model, conf_thresh):
    arduino_ip = arduino_ip  # IP-Adresse des Arduino
    port = port  # Muss mit der Portnummer im Arduino-Sketch übereinstimmen

    # Socket erstellen
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((arduino_ip, port))

    received_values = []  # Liste zum Speichern der empfangenen Werte

    # Pfad zum Speicherordner
    save_path = 'daten2/obj'

    # Überprüfen, ob der Speicherordner existiert, andernfalls erstellen
    os.makedirs(save_path, exist_ok=True)
    model = YOLO(model)

    def classify_image(filename):
        results = model.predict(source=filename, conf=conf_thresh)
        # Ergebnisse extrahieren (Liste von Result-Objekten, hier nur 1 Bild → results[0])
        result = results[0]
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])  # Klassenindex (z. B. 0 für „ball“)
            label = model.names[cls_id]  # Klassennamen (z. B. "ball")
            conf = float(box.conf[0])  # Konfidenzscore
            xyxy = box.xyxy[0].tolist()  # Bounding Box Koordinaten [x1, y1, x2, y2]

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": xyxy
            })
        print(detections)

        annotated_img = result.plot()
        annotated_pil = Image.fromarray(annotated_img)
        annotated_pil.save('annotated_latest.jpg')

        # 3 Sekunden warten
        if detections:
            # Nur das erste erkannte Objekt verwenden
            det = detections[0]
            label = det['label']
            conf = round(det['confidence'], 2)
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
        else:
            # Kein Objekt erkannt
            label = "none"
            conf = 0
            x1 = y1 = x2 = y2 = 0

        # Arduino-freundlicher, einfacher CSV-String
        result_str = f"{label},{conf},{x1},{y1},{x2},{y2}\n"
        return result_str

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
                    if (value == 42):
                        image_path = capture_image(url)
                        if image_path:
                            result_str = classify_image(image_path)
                            client_socket.sendall(result_str.encode())

                except ValueError:
                    print(f"Fehler beim Konvertieren: {response}")  # Falls ungültige Daten empfangen werden

            # Falls du eine Bedingung zum Stoppen willst (z. B. nach 10 Werten)
            if len(received_values) >= 10:
                print("Genug Werte empfangen, beende die Verbindung.")
                break

    finally:
        client_socket.close()


