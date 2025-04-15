#include <WiFi.h>

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
}

void loop() {
  WiFiClient client = server.available();
  if (client) {
    Serial.println("Client verbunden.");
    while (client.connected()) {
      if (client.available()) {
        String data = client.readStringUntil('\n');
        Serial.print("Empfangene Daten: ");
        Serial.println(data);
        if(data=="1"){
  digitalWrite(LED_RED, LOW);   // LOW schaltet die LED ein
  delay(500);                   // Warte 500 Millisekunden
  digitalWrite(LED_RED, HIGH);  // HIGH schaltet die LED aus

  // Grüne LED einschalten
  digitalWrite(LED_GREEN, LOW);
  delay(500);
  digitalWrite(LED_GREEN, HIGH);

  // Blaue LED einschalten
  digitalWrite(LED_BLUE, LOW);
  delay(500);
  digitalWrite(LED_BLUE, HIGH);       
        }else{
          delay(10);
        }
        // Hier können Sie die empfangenen Daten weiterverarbeiten
      }
    }
    client.stop();
    Serial.println("Client getrennt.");
  }
}
