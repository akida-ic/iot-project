# Indoor Environment & Sleep Quality — IoT Sensing Project

**ELEC70126 · Imperial College London · Feb 2026**

This project investigates the relationship between indoor environmental conditions (daylighting, humidity, temperature) and sleep quality using a low-cost IoT sensing system.

**Dashboard:** https://iot-project-asay.onrender.com

---

## Repository Structure

```
iot-project/
├── esp32_firmware/
│   └── data_collector.ino   # ESP32 sensor firmware (Arduino)
├── app_dash.py              # Interactive Plotly Dash dashboard
├── data_prep.py             # Data loading, cleaning, lag merging
├── analysis.py              # Statistical analysis & figure generation
├── garmin_fetch.py          # Garmin Connect data collection
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Hardware

| Component | Role |
|-----------|------|
| Heltec WiFi LoRa 32 V3 (ESP32) | Microcontroller |
| LDR (via voltage divider, ADC pin 4) | Indoor light intensity |
| DHT11 (pin 2) | Temperature & humidity |

---

## ESP32 Firmware (`esp32_firmware/data_collector.ino`)

Samples LDR, temperature, and humidity every 15 minutes. Data is saved locally to SPIFFS (`/data.csv`) and uploaded to Google Sheets via a Google Apps Script HTTP endpoint.

### Configuration

Before flashing, set your credentials in the firmware:

```cpp
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";
const char* SCRIPT_URL = "YOUR_GOOGLE_APPS_SCRIPT_URL";
```

### Dependencies (Arduino Library Manager)
- `DHT sensor library` by Adafruit
- `HTTPClient` (built-in ESP32)
- `SPIFFS` (built-in ESP32)

### Serial Commands
- Send `U` via Serial Monitor to manually trigger an upload

---

## Python Setup

```bash
pip install -r requirements.txt
```

### Configuration

Create a `config.py` file in the project root before running any scripts:

```python
GARMIN_EMAIL = "your@email.com"
GARMIN_PASSWORD = "yourpassword"
LAT = 51.5074
LON = -0.1278
DB = "data/iot_data.db"
```

> Note: `config.py` is excluded from version control. Never commit credentials to the repository.

### Run Garmin data collection
```bash
python garmin_fetch.py
```

### Run analysis
```bash
python analysis.py
```
Outputs figures to `figures/`.

### Run dashboard locally
```bash
python app_dash.py
```
Opens at `http://localhost:8050`

---

## Data Pipeline

```
ESP32 → Google Sheets → SQLite → indoor_data.xlsx
Open-Meteo API → weather_15min.xlsx
Garmin Connect API → garmin_fetch.py → SQLite → CSV/Excel
```

All data files are merged in `data_prep.py` into a single DataFrame (n = 13 days).

---

## Data Files Required

Place the following in the project root:

| File | Source |
|------|--------|
| `indoor_data.xlsx` | ESP32 via Google Sheets |
| `weather_15min.xlsx` | Open-Meteo API |
| `sleep_summary.xlsx` | Garmin Connect |
| `sleep_stages.csv` | Garmin Connect |
| `sleep_hr_timeseries.csv` | Garmin Connect |
| `sleep_respiration_timeseries.csv` | Garmin Connect |

---

## Key Findings

- **Solar → Indoor brightness:** r = +0.832, p < 0.001 (daily means, n = 13)
- **Sleep-period humidity → Sleep score:** r = −0.527, p = 0.078 (n = 12 nights)
- **Brightness → Deep sleep %:** r = −0.477, p = 0.099, attenuated by outdoor humidity confounding
- **Indoor temperature:** no significant association with any sleep outcome

---

## Deployment

The dashboard is deployed on Render (free tier):

- **Start command:** `gunicorn app_dash:server`
- **Build command:** `pip install -r requirements.txt`

Note: free tier instances spin down after 15 minutes of inactivity. Allow 30–60 seconds for cold start.