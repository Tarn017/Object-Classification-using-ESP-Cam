#include <WiFi.h>

const int buttonPin = 2;     // the number of the pushbutton pin

// variables will change:
int buttonState = 0;         // variable for reading the pushbutton status

// WLAN-Zugangsdaten
const char* ssid = "TP-Link_92AC";
const char* password = "73125785";

const uint16_t port = 12345; // Portnummer für die Verbindung

WiFiServer server(port);

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_BLUE, OUTPUT);

  Serial.print("Verbinde mit dem WLAN...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("\nVerbunden!");
  Serial.print("IP-Adresse: ");
  Serial.println(WiFi.localIP());

  server.begin();
  Serial.println("Server gestartet, wartet auf Verbindungen...");
  pinMode(buttonPin, INPUT);
}

void loop() {
  WiFiClient client = server.available();
  if (client) {
    Serial.println("Client verbunden.");
    while (client.connected()) {
          buttonState = digitalRead(buttonPin);

    if (buttonState == HIGH) {
      int i = 42;
      Serial.println("pressed");
      client.println(i);
      delay(500);
    }
      if (client.available()) {
        String data = client.readStringUntil('\n');
        Serial.print("Empfangene Daten: ");
        Serial.println(data);
int i1 = data.indexOf(',');
int i2 = data.indexOf(',', i1 + 1);
int i3 = data.indexOf(',', i2 + 1);
int i4 = data.indexOf(',', i3 + 1);
int i5 = data.indexOf(',', i4 + 1);
int i6 = data.indexOf(',', i5 + 1);  // optional: falls du später mehr willst

String x1_str = data.substring(i2 + 1, i3);
String y1_str = data.substring(i3 + 1, i4);
String x2_str = data.substring(i4 + 1, i5);
String y2_str = data.substring(i5 + 1);

int x1 = x1_str.toInt();
int y1 = y1_str.toInt();
int x2 = x2_str.toInt();
int y2 = y2_str.toInt();
int centerX = (x1 + x2) / 2;
int centerY = (y1 + y2) / 2;
Serial.print(centerX);
Serial.print(",");
Serial.println(centerY);
if(centerX!=0){
  if (centerX > 160 && centerY < 120) {
    Serial.println("Flasche ist rechts oben");
  }
  else if (centerX <= 160 && centerY < 120) {
    Serial.println("Flasche ist links oben");
  }
  else if (centerX <= 160 && centerY >= 120) {
    Serial.println("Flasche ist links unten");
  }
  else if (centerX > 160 && centerY >= 120) {
    Serial.println("Flasche ist rechts unten");
  }
}

      }
    }
    client.stop();
    Serial.println("Client getrennt.");
  }
}
