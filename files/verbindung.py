import socket

def check_connection(ip, port, timeout=5):
    try:
        with socket.create_connection((ip, port), timeout):
            print(f"Verbindung zu {ip}:{port} erfolgreich.")
            return True
    except (socket.timeout, socket.error) as ex:
        print(f"Keine Verbindung zu {ip}:{port}. Fehler: {ex}")
        return False

if __name__ == "__main__":
    esp32_ip = "172.20.10.8"  # IP-Adresse der ESP32-CAM
    esp32_port = 81           # Standardport für den Videostream

    check_connection(esp32_ip, esp32_port)
