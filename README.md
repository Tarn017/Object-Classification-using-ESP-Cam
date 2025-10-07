# Computer Vision using ESP-Cam

**Material:** ESP32-CAM, ESP32_CAM-adapter, Arduino Nano ESP32, Wi-Fi router
**Software:** Arduino, Python

**Einrichten der EspCam:**

1.	EspCam mit Adapter verbinden
2.	esp32 by Espressif im Boardmanager installieren
3.	Öffne: File -> Examples -> Esp32 -> Camera -> CameraWebServer
4.	Ergänze Wlan-Daten in CameraWebServer.ino
5.	In der Leiste oben auf board_config.h wechseln
6.	#define CAMERA_MODEL_WROVER_KIT mit // auskommentieren
7.	Vor #define CAMERA_MODEL_AI_THINKER  // entfernen
8.	Zurück in CameraWebServer.ino passenden Port auswählen und als Board unter esp32 "AI Thinker Esp32-Cam"

# Object Classification 

**Download**

Download the following script: [obj](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/obj.py). It contains all relevant functions.

Arduino-Code:  [script](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/NanoEsp_classification.ino)

**Explanation**

Import the following functions into your script: `from obj import abfrage, training_classification, testen_classification, neural_network_classification`

A detailed explanation of the functions and other things can be found here: [Object_Classification](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/Klassifizierung%20im%20CNN.docx)

**How to use the functions:**

`aufnahme(url, interval, klasse, ordner)`: Serves the purpose of data collection. 'url' is the url of the ESPcam. It's printed on the Serial Monitor of the Arduino IDE when you upload the arduino code. 'interval' refers to the amount of pictores taken. `interval=0.5` is euivalent to a picture taken every 0.5 seconds. 'klasse' refers to the name of the object you're taking the pictures of.

`training_classification(ordner, model_name, epochen, n_full, pool_size, conv_filter, filter_size, droprate, resize padding, val_ordner, aug_parameter, alpha)`: Function to train the neural network. It automatically generates and trains a neural Network with the chosen parameters and saves it as 'model_name'.

`validation_classification(model, val_ordner, padding)`: Given validation data in a directory with the same structure as the directory the model was trained on. With this function, the model can be tested on this data.

`result, confidence = testen_classification(url, model, ordner, padding, live, interval)`: Takes a picture and classifies it using the model <model_name>. If live=True, it takes a picture every <interval> second and classifies it until the program is terminated.

`neural_network_classification(url, arduino_ip, ordner, port, model)`: Establishes a connection to the ESPcam as well as to the Arduino Nano esp. If it receives the command from the arduino, a picture is taken and classified afterwards. The result is sent back to the Arduino. 'arduino_ip' is printed on the Serial Monitor of the arduino IDE when the according [code](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/NanoEsp_classification.ino) is uploaded. The port needs to be defined in the code, default=12345.

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
aufnahme(url, 
         interval, 
         klasse, 
         ordner)  # Start data collection

training_classification(ordner,
                        model_name,
                        epochen,
                        n_full = [512],
                        pool_size = 2,
                        conv_filter=[32, 64, 128, 256],
                        filter_size=3,
                        droprate=0.5,
                        resize=[160,160],
                        padding=True,
                        val_ordner='daten3val',
                        aug_parameter=['vertical', 0.6,0.05,0.1],
                        alpha=0.001)   # Train a neural network for image classification

Validation_classification(model_name,
                          val_ordner="daten3val",
                          padding=True)  #Test neural Network on collected data

result, confidence = testen_classification(url, 
                      model_name, 
                      ordner, 
                      padding=False, 
                      live=True, 
                      interval=5)  # Capture an image with the camera every 5 seconds and classify it; result = class, confidence = probability
print(result)

neural_network_classification(url, arduino_ip, ordner, port, model_name)
```

# Object detection

**Download**

Download the following script: [obj](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/obj.py). It contains all relevant functions.

Arduino-Code: [script]( https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/NanoEsp.ino)

**Explanation**

Import the following functions into your script: `from obj import abfrage, training_detection, testen_detection, neural_network_detection`

A detailed explanation of the functions and other things can be found here: [Object_Detection](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/Detection%20im%20CNN.docx)

**How to use the functions:**

`aufnahme(url, interval, klasse, ordner)`: Serves the purpose of data collection. 'url' is the url of the ESPcam. It's printed on the Serial Monitor of the Arduino IDE when you upload the arduino code. 'interval' refers to the amount of pictores taken. `interval=0.5` is euivalent to a picture taken every 0.5 seconds. 'klasse' refers to the name of the object you're taking the pictures of. However, the class is not relevant in case of object detection as the pictures are annotated in the next step.

Label your collected data. For example, you can use *roboflow* (recommended). Use YOLO v8 format. Create the dataset and download it via code. Example:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="key")
project = rf.workspace("workspace").project("bottlez")
version = project.version(2)
```

`training_detection(dataset, epochen, img_size)`: Function to train the neural network. It uses YOLO v8 to create a detection model.

`result = testen_detection(url, model, conf_thresh, img_size)`: Takes a picture and and performs object detection. 'model' refers to the path of the trainied model, usually something like `model = 'runs/detect/train/weights/best.pt'`. 'conf_thesh' refers to the treshold of detection. Is the result of the dtection lower, then no object is detected. Value between 0 and 1.

`neural_network_detection(url, arduino_ip, port, model, conf_thresh)`: Establishes a connection to the ESPcam as well as to the Arduino Nano esp. If it receives the command from the arduino, a picture is taken and object detection is performed afterwards. The result is sent back to the Arduino. 'arduino_ip' is printed on the Serial Monitor of the arduino IDE when the according code is uploaded. The port needs to be defined in the code itself.

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




