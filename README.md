# Object-Classification-using-ESP-Cam

**Material:** ESP32-CAM, Serial to USB adapter, Arduino Nano ESP32, Wi-Fi router
**Software:** Arduino, Python

**1**
Set up the ESP32-CAM following this guide:
[Getting Started With ESP32-CAM: A Beginner's Guide](https://randomnerdtutorials.com/getting-started-with-esp32-cam/)
Connect the ESP32-CAM to the adapter and connect the adapter to the laptop.
To boot, connect IO0 to GND, then disconnect this connection and press the RESET button.
Only the Wi-Fi router information needs to be adjusted.
Afterwards, an IP address will be displayed on the Serial Monitor.
You can capture an image via browser or Python using <IP>/capture in your browser.

**2**
Next is a script that captures and saves an image every half second (for training data).
In theory, only the class name needs to be adjusted here and the frequency of pictures taken.
[Script](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/abfrage.py)

**3**
With the collected data, a neural network is trained.
The architecture can be chosen freely.
[Script](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/CNN.py)

**4**
The trained network can then be used for predictions (classification).
It is important to adjust the class names and, in particular, bring them into the order in which the folders were stored.
[Script](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/klassifizierung.py)

**5**
Below is an automated classification script.
It receives a number from the Arduino.
If this number matches a specific value, classification is triggered.
This way, classification is only performed when, for example, a button is pressed.
The result of the classification is then sent back to the Arduino.
[Script](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/sendenEmpfangen.py)

**6**
Below is the script for the Arduino.
It currently works so that it sends a signal when a button is pressed.
Then a classification is triggered, and the Arduino receives the result.
With if-conditions, it reacts accordingly.
[Script](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/wifi.ino)


# Object Classification 

**Download**

Download the following script: [obj](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/wifi.ino). It contains all relevant functions.

**Explanation**

Import the functions into your script: `from obj import abfrage, training_classification, testen_classification, neural_network_classification`

**How to use the functions:**

`abfrage(url, interval, klasse)`: Serves the purpose of data collection. 'url' is the url of the ESPcam. It's printed on the Serial Monitor of the Arduino IDE when you upload the arduino code. 'interval' refers to the amount of pictores taken. `interval=0.5` is euivalent to a picture taken every 0.5 seconds. 'klasse' refers to the name of the object you're taking the pictures of.

`training_classification(ordner, model_name, epochen)`: Function to train the neural network. 'ordner' refers to the directory where the pictures are saved, defaul: ordner='daten'.'model_name': Choose a name for your model with the ending .keras. 'epochen' equals the numer of training epochs.

`result, confidence = testen(url, model_name, ordner)`: Takes a picture and classifies it using the model <model_name>. 

`neural_network_classification(url, arduino_ip, ordner, port)`: Establishes a connection to the ESPcam as well as to the Arduino Nano esp. If it receives the command from the arduino, a picture is taken and classified afterwards. The result will be sent back to the Arduino. 'arduino_ip' is printed on the Serial Monitor of the arduino IDE when the according code is uploaded. The port needs to be defined in the code itself.

**Example**
```python
url = 'http://192.168.1.100/capture' #URL der ESPcam -> Diese wird direkt auf dem Serial Monitor von Arduino ausgegeben
ordner = 'daten' #Den Order definieren, in dem die Unterordner mit den Bildern liegen (Standardmäßig in "daten") (Object Classification)
model_name = "objekt_klassifikator.keras" #Definieren des Namens des Modells !muss mit .keras enden! (Object Classification)
epochen = 15 #Anzahl der Trainingsepochen
interval = 0.5 #Intervall in dem die Bilder aufgenommen werden. Hier: alle 0.5 Sekunden
arduino_ip = '192.168.1.102' #IP des Arduino Nano Esp. Wird in arduino ausgegeben
port = 12345 #Port des Arduinos. Wird im Arduino-Code festgelegt

klasse = 'schachtel' #für welche Klasse sollen Daten gesammelt werden?
abfrage(url, interval, klasse) #Start des Daten sammelns

training_classification(ordner, model_name, epochen) #Training eines Neuronalen Netzes zur Bildklassifizierung

result, confidence = testen(url, model_name, ordner) #Es wird ein Bild mit der cam aufgenommen und klassifiziert. result entspricht Klasse und conf der Wahrscheinlichkeit
print(result)

neural_network_classification(url, arduino_ip, ordner, port)





