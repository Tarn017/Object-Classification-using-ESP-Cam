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
[Script](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/abfrage.py)

**4**
The trained network can then be used for predictions (classification).
It is important to adjust the class names and, in particular, bring them into the order in which the folders were stored.
[Script](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/abfrage.py)

**5**
Below is an automated classification script.
It receives a number from the Arduino.
If this number matches a specific value, classification is triggered.
This way, classification is only performed when, for example, a button is pressed.
The result of the classification is then sent back to the Arduino.

**6**
Below is the script for the Arduino.
It currently works so that it sends a signal when a button is pressed.
Then a classification is triggered, and the Arduino receives the result.
With if-conditions, it reacts accordingly.
