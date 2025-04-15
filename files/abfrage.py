import requests
import time
import os

# URL des ESP32-CAM /capture Endpunkts
url = 'http://192.168.1.100/capture'

# Intervall zwischen den Aufnahmen in Sekunden
interval = 0.1

# Pfad zum Speicherordner
save_path = 'daten/Tobias'

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
