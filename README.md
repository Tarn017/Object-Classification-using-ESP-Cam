# Computer Vision using ESP-Cam

**Material:** ESP32-CAM, ESP32_CAM-adapter, Arduino Nano ESP32, Wi-Fi router
**Software:** Arduino, Python

**Steps:**  
1. EspCam einrichten
2. Daten sammeln  
3. Neuronales Netz trainieren  
4. Neuronales Netz testen  
5. Arduino Microcontroller einrichten  
6. Trainiertes Netz auf Microcontroller deployen  

# 1. Einrichten der EspCam

1.	EspCam auf Adapter stecken
2.	In der Arduino IDE: Board-Manager öffnen über Tools -> Board -> Boards Manager
3.	esp32 by Espressif im Boardmanager installieren (Über Suchleiste oben suchen)
4.	In Arduino IDE: File -> Examples -> Esp32 -> Camera -> CameraWebServer
5.	Ergänze Wlan-Daten in CameraWebServer.ino
6.	In der Tableiste oben gegebenenfalls auf board_config.h wechseln (Falls die `#define`-Einstellungen nicht direkt in dieser Datei getroffen werden) 
7.	Vor `#define CAMERA_MODEL_WROVER_KIT` die Striche // hinzufügen
8.	Vor `#define CAMERA_MODEL_AI_THINKER`  // entfernen
9.	Zurück in CameraWebServer.ino passenden Port auswählen und als Board unter esp32 "AI Thinker Esp32-Cam"
10.	Code uploaden
11.	Auf dem Serial Monitor wir nun eine URL ausgegeben:

![cam_url](https://raw.githubusercontent.com/Tarn017/Object-Classification-using-ESP-Cam/main/assets/cam_url.png)

12. Kopiere die URL nun in einen Brrowser der mit demselben WLAN, verbundne ist, wie die Kamera, um zu sehen ob alles korrekt funktioniert hat

Eine genauere Beschreibung der einzelnen Schritte findet ihr unter [Getting Started With ESP32-CAM](https://lastminuteengineers.com/getting-started-with-esp32-cam/)

# 2. Daten Sammeln

**Einrichten:**  
Erstellt in eurer Python IDE ein neues Projekt und legt ein nues Haupt-File an (bspw. main.py).  
Ladet das folgende Skript herunter und kopiert es in euer Python-Projekt sodass es direkt neben eurem Haupt-File zu sehen ist: [project.py](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/src/project.py)

![project.py Screenshot](https://raw.githubusercontent.com/Tarn017/Object-Classification-using-ESP-Cam/main/assets/project_py.png)

**Abhängigkeiten installieren:**
Lade die folgende Datei herunter und kopiere sie in deinen Projektordner: [requirements_class.txt](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/src/requirements_class.txt)

Gehe anschließend in deiner IDE unten auf das Terminal und installiere alle Pakete mit `pip install -r requirements_class.txt`:

![requirements Screenshot](https://raw.githubusercontent.com/Tarn017/Object-Classification-using-ESP-Cam/main/assets/requirements.png)

**Los gehts:**  
Beim einrichten der EspCam wurde eine URL ausgegeben. Diese wird für die nächsten Schritte benötigt.  
Geht in euer Hauptskrip und fügt ganz oben die Zeile `from project import aufnahme` ein. Als nächstes fügt ihr darunter `if __name__ == "__main__":` ein. Alles was hiernah kommt muss nach rechts eingerückt werden. Nun kann `aufnahme` genutzt werden um Bilddaten zu sammeln. Sie nimmt alle paar Sekunden ein Bild auf und speichert dieses automatisch in einer benannten Klasse. Die Funktion ist folgendermaßen aufgebaut:  
`aufnahme(url, interval, klasse, ordner)`: *url* entspricht der URL von der EspCam, diese kann einfach kopiert werden. Wichtig ist nur, dass sie in Anführungszeichen steht. *interval* entspricht der Frequenz, in der Bilder aufgenommen werden (1 bspw. für alle 1 Sekunden). *klasse* sollte dem Klassenname entsprechen, für dessen Klasse gerade Daten gesammelt werden. Für *ordner* kann ebenfalls ein beliebiger Name gewählt werden. Es wird ein Ordner mit selbigem Namen automatisch erstellt in dem die Klassen und Bilder gespeichert werden.

Hier ein Beispiel. Wichtig ist, dass Anführungszeichen übernommen werden, dort wo sie gebraucht werden:  
```python
from project import CNN, aufnahme, testen_classification, neural_network_classification, FFN

if __name__ == "__main__":
    aufnahme(url='http://172.20.10.3', interval=1, klasse='noise', ordner='Objekte')
```

Zum starten kann nun einfach das Skript ausgeführt werden. Für jede Klasse, für die Daten gesammelt werden sollen, muss das Skript separat ausgeführt werden. Ein umbenennen von *klasse* ist während das Skript läuft nicht möglich.

# 3. Neuonales Netz trainieren

Um ein Modell zu trainieren, muss nun zusätzlich ein Trainingsskript am Anfang importiert werden: `from project import CNN`. Konkret handelt es sich hierbei um ein Convolutional Neural Network, das sich flexibel einstellen lässt. Wichtig ist, dass die Funktion `CNN()` ebenfalls wieder eingerückt unter `if __name__ == "__main__":` steht. Hier eine kurze Erklärung, wie man die Funktion nutzt:  
`CNN(train_path, epochs, lr, conv_filters, fully_layers, resize, model_name, train_split, droprate, augmentation, dec_lr)`  
1. *train_path* entspricht dem Ordner auf den ihr das Netz trainieren wollt
2. *epochs* entspricht der Anzahl an Epochen, die das Netz trainiert werden soll
3. *lr* entspricht der Lernrate
4. *conv_filters* hat die Form [x,y,z,...], wobei x der Anzahl an convolutional Filtern in der 1. Schicht entsprich, y der Anzahl in der 2., usw. Die GEsamtzahl an Schichten wird somit ebenfalls hier bestimmt.
5. *fully_layers* hat die Form [x,y,z,...], wobei x der Anzahl an Neuronen in der 1. voll verbundenen Schicht entspricht, usw.
6. *resize* hat die form (höhe,breite), wobei beide Angaben in Anzahl Pixel gemacht werden. Quadratische Angaben werden bevorzugt
7. *model_name* gibt eurem Modell einen Namen. Beachte: Existiert bereits eines mit demselben Namen, wird das alte überschrieben.
8. *train_split* bspw. 0.8 bedeutet, dass 80% der Bilder fürs Training und 20% für Validation genutzt werden. Bei 1 werden alle Daten für das Training genutzt.
9. *droprate* (optional) bspw 0.2 bedeutet, dass jedes Neuron mit einer Wahrscheinlichkeit von 20% deaktiviert wird.
10. *augmentation* (optional) hat die Form [flip, rotate, brightness, contrast, saturation]. Jeder dieser Werte muss zwischen 0 und 1 liegen. *flip* ist die Wahrscheinlichkeit, dass ein Bild gespiegelt wird, *rotate* gibt die Stärke einer zufälligen Rotation an (1 für stark), *brightness, contrast, saturation* geben an wie stark Helligkeit, Kontrast und Sättigung maximal verändert werden.
11. *dec_lr* (optional) wirg genutzt, falls die Lernrate während des Trainigs abnehmen soll. Der Wert der für diesen Parameter angegeben wird, entspricht der Lernrate der letzten Epoche.

Hier ein Beispiel. Wichtig ist, dass Anführungszeichen übernommen werden, dort wo sie gebraucht werden und der Datentyp für jedem Parameter richtig gewählt ist: 
```python
from project import CNN

if __name__ == "__main__":
    CNN(
        train_path="zml_klass",
        epochs=15,
        lr=0.001,
        conv_filters=[16, 32, 64, 128],
        fully_layers=[256],
        resize=(128, 128),
        model_name='peter',
        train_split=0.9,
        droprate=0,
        augmentation=[0, 0, 0, 0, 0],
        dec_lr=0.001
    )
```
# 4. Neuronales Netz testen

Um das Modell zu testen muss zusätzlich oben das entsprechende Skript importiert werden: `from project import testen_classification`. Diese nimmt live Bilder mit der EspCam auf und klassifiziert diese im Anschluss direkt.  
`testen_classification(url, model_name, live, interval)`:  
*url* entspricht wieder der URL der EspCam.
*model_name* entspricht dem Namen des Modells welches getestet werden soll.
*live* ist gleich True oder False. Wenn False ausgewählt ist, wird nur ein einziges Bild aufgenommen und klassifiziert.
*interval* ist die Dauer zwischen zwei Aufnahmen in Sekunden, falls True ausgewählt ist.
```python
from project import CNN

if __name__ == "__main__":
    testen_classification(url='http://172.20.10.3', 
                              model_name='peter', 
                              live=True, 
                              interval=3)
```
