#include <ESP8266WiFi.h>
#include <WebSocketsClient.h>

// Replace with your network credentials
const char* ssid = "Your_SSID";
const char* password = "Your_PASSWORD";

// Server details
const char* serverIP = "192.168.1.100"; // Replace with your server IP
const int serverPort = 8080; // Replace with your server port

WebSocketsClient webSocket;
WiFiClient client;

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("Connected to WiFi");

  webSocket.begin(serverIP, serverPort, "/ws");
  webSocket.onMessage(onMessage);
}

void loop() {
  webSocket.loop();
}

void onMessage(WebsocketsMessage message) {
  String payload = message.data;
  if (payload == "unlock") {
    unlockDoor();
  }
}

void unlockDoor() {
  // Replace with actual code to control the door lock
  Serial.println("Door unlocked!");
}
