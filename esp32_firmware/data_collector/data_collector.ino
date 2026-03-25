#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include "SPIFFS.h"
#include "time.h"

#define LDR_PIN 4
#define DHT_PIN 2
#define DHT_TYPE DHT11

const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";
const char* SCRIPT_URL = "YOUR_GOOGLE_APPS_SCRIPT_URL";
const char* NTP_SERVER = "pool.ntp.org";
const long  GMT_OFFSET = 0;
const int   DAYLIGHT_OFFSET = 0;

DHT dht(DHT_PIN, DHT_TYPE);
unsigned long lastSample = 0;
const unsigned long interval = 900000;  // 15 minutes
bool uploading = false;
bool timeReady = false;

/* ---------- Upload Index ---------- */
int readUploadIndex() {
  if (!SPIFFS.exists("/upload_index.txt")) return 0;
  File f = SPIFFS.open("/upload_index.txt", FILE_READ);
  if (!f) return 0;
  int val = f.parseInt();
  f.close();
  return val;
}

void writeUploadIndex(int val) {
  File f = SPIFFS.open("/upload_index.txt", FILE_WRITE);
  if (f) { f.print(val); f.close(); }
}

/* ---------- WiFi ---------- */
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(500);
  }
}

/* ---------- Time ---------- */
void syncTime() {
  if (WiFi.status() != WL_CONNECTED) return;
  configTime(GMT_OFFSET, DAYLIGHT_OFFSET, NTP_SERVER);
  struct tm t;
  int retry = 0;
  while (!getLocalTime(&t) && retry < 15) { delay(1000); retry++; }
  if (retry < 15) timeReady = true;
}

String getTimestamp() {
  struct tm t;
  if (getLocalTime(&t)) {
    char buf[25];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &t);
    return String(buf);
  }
  return "";
}

/* ---------- Upload ---------- */
void uploadData() {
  uploading = true;
  connectWiFi();
  syncTime();
  if (!SPIFFS.exists("/data.csv")) { uploading = false; return; }
  File file = SPIFFS.open("/data.csv", FILE_READ);
  if (!file) { uploading = false; return; }

  HTTPClient http;
  http.setTimeout(10000);
  http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);

  int uploadedIndex = readUploadIndex();
  int currentLine = 0;

  while (file.available()) {
    String line = file.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) continue;
    currentLine++;
    if (currentLine <= uploadedIndex) continue;

    int p1 = line.indexOf(',');
    int p2 = line.indexOf(',', p1 + 1);
    int p3 = line.indexOf(',', p2 + 1);
    if (p1 < 0 || p2 < 0 || p3 < 0) continue;

    String ts    = line.substring(0, p1);
    String temp  = line.substring(p1 + 1, p2);
    String hum   = line.substring(p2 + 1, p3);
    String light = line.substring(p3 + 1);

    String payload = "{\"timestamp\":\"" + ts +
                     "\",\"temperature\":\"" + temp +
                     "\",\"humidity\":\"" + hum +
                     "\",\"light\":\"" + light + "\"}";

    http.begin(SCRIPT_URL);
    http.addHeader("Content-Type", "application/json");
    int response = http.POST(payload);
    Serial.print("Line "); Serial.print(currentLine);
    Serial.print(" HTTP code: "); Serial.println(response);
    http.end();

    if (response != -1) writeUploadIndex(currentLine);
    delay(300);
  }
  file.close();
  uploading = false;
}

/* ---------- Setup ---------- */
void setup() {
  Serial.begin(115200);
  delay(1000);
  SPIFFS.begin(true);
  dht.begin();
  connectWiFi();
  syncTime();
  lastSample = millis();
}

/* ---------- Loop ---------- */
void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "U") uploadData();
  }

  if (!timeReady) { connectWiFi(); syncTime(); }

  if (!uploading && timeReady && millis() - lastSample >= interval) {
    lastSample = millis();
    int   ldr  = analogRead(LDR_PIN);
    float temp = dht.readTemperature();
    float hum  = dht.readHumidity();

    if (!isnan(temp) && !isnan(hum)) {
      String ts = getTimestamp();
      if (ts == "") return;
      String line = ts + "," + String(temp, 1) + "," +
                    String(hum, 1) + "," + String(ldr);
      File file = SPIFFS.open("/data.csv", FILE_APPEND);
      if (file) { file.println(line); file.close(); }
      Serial.println("Saved: " + line);
    }
  }
}
