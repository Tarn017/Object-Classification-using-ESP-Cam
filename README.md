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

Download the following script: [obj](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/obj.py). It contains all relevant functions.
Arduino-Code:  [script](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/NanoEsp_classification.ino)
**Explanation**

Import the functions into your script: `from obj import abfrage, training_classification, testen_classification, neural_network_classification`

**How to use the functions:**

`abfrage(url, interval, klasse)`: Serves the purpose of data collection. 'url' is the url of the ESPcam. It's printed on the Serial Monitor of the Arduino IDE when you upload the arduino code. 'interval' refers to the amount of pictores taken. `interval=0.5` is euivalent to a picture taken every 0.5 seconds. 'klasse' refers to the name of the object you're taking the pictures of.

`training_classification(ordner, model_name, epochen)`: Function to train the neural network. 'ordner' refers to the directory where the pictures are saved, defaul: ordner='daten'.'model_name': Choose a name for your model with the ending .keras. 'epochen' equals the numer of training epochs.

`result, confidence = testen(url, model_name, ordner)`: Takes a picture and classifies it using the model <model_name>. 

`neural_network_classification(url, arduino_ip, ordner, port)`: Establishes a connection to the ESPcam as well as to the Arduino Nano esp. If it receives the command from the arduino, a picture is taken and classified afterwards. The result is sent back to the Arduino. 'arduino_ip' is printed on the Serial Monitor of the arduino IDE when the according code is uploaded. The port needs to be defined in the code itself.

**Example**
```python
url = 'http://192.168.1.100/capture' # URL of the ESP32-CAM -> This is displayed directly in the Arduino Serial Monitor
ordner = 'daten' # Define the folder where the subfolders with the images are located (default: "daten") (Object Classification)
model_name = "objekt_klassifikator.keras" # Define the name of the model; must end with .keras (Object Classification)
epochen = 15 # Number of training epochs
interval = 0.5 # Interval at which images are captured; here: every 0.5 seconds
arduino_ip = '192.168.1.102' # IP address of the Arduino Nano ESP; displayed in Arduino Serial Monitor
port = 12345 # Port of the Arduino; defined in the Arduino code

klasse = 'schachtel' # For which class should data be collected?
abfrage(url, interval, klasse) # Start data collection

training_classification(ordner, model_name, epochen) # Train a neural network for image classification

result, confidence = testen(url, model_name, ordner) # Capture an image with the camera and classify it; result = class, confidence = probability
print(result)

neural_network_classification(url, arduino_ip, ordner, port)
```

# Object detection
**Download**

Download the following script: [obj](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/obj.py). It contains all relevant functions.
Arduino-Code: [script]( https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/NanoEsp.ino)

**Explanation**

Import the functions into your script: `from obj import abfrage, training_detection, testen_detection, neural_network_detection`

**How to use the functions:**

`abfrage(url, interval, klasse)`: Serves the purpose of data collection. 'url' is the url of the ESPcam. It's printed on the Serial Monitor of the Arduino IDE when you upload the arduino code. 'interval' refers to the amount of pictores taken. `interval=0.5` is euivalent to a picture taken every 0.5 seconds. 'klasse' refers to the name of the object you're taking the pictures of.

Label your collected data. For example, you can use *roboflow* (recommended). Use YOLO v8 format. Create the dataset and download it via code. Example:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="key")
project = rf.workspace("workspace").project("bottlez")
version = project.version(2)
```

`training_detection(version, epochen)`: Function to train the neural network. It uses YOLO v8 to create a detection model. 'epochen' equals the numer of training epochs.

`result = testen_detection(url, model, conf_thresh = 0.2)`: Takes a picture and and performs object detection. 'model' refers to the path of the trainied model, usually something like `model = 'runs/detect/train/weights/best.pt'`. 'conf_thesh' refers to the treshold of detection. Is the result of the dtection lower, then no object is detected. Something between 0 and 1.

`neural_network_detection(url, arduino_ip, port, model, conf_thresh=0.1)`: Establishes a connection to the ESPcam as well as to the Arduino Nano esp. If it receives the command from the arduino, a picture is taken and object detection is performed afterwards. The result is sent back to the Arduino. 'arduino_ip' is printed on the Serial Monitor of the arduino IDE when the according code is uploaded. The port needs to be defined in the code itself.

**Example**
```python
url = 'http://192.168.1.100/capture' # URL of the ESP32-CAM -> This is displayed directly in the Arduino Serial Monitor
epochen = 15 # Number of training epochs
interval = 0.5 # Interval at which images are captured; here: every 0.5 seconds
arduino_ip = '192.168.1.102' # IP address of the Arduino Nano ESP; displayed in Arduino Serial Monitor
port = 12345 # Port of the Arduino; defined in the Arduino code

from roboflow import Roboflow
rf = Roboflow(api_key="Zbo6tBzjKmXWSq7ndLDS")
project = rf.workspace("karlsruher-institut-fr-technologie-7bdnc").project("bottlezml")
version = project.version(2)

training_detection(version, epochen)
model = 'runs/detect/train/weights/best.pt'

result = testen_detection(url, model, conf_thresh = 0.2)
print(result)

neural_network_detection(url, arduino_ip, port, model, conf_thresh=0.1)
```




