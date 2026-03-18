import sqlite3
import requests
from datetime import datetime, timedelta
from config import DB, LAT, LON

conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS weather_15min (
    datetime TEXT PRIMARY KEY,
    cloud_cover REAL,
    temperature REAL,
    humidity REAL,
    solar_radiation REAL,
    direct_radiation REAL,
    diffuse_radiation REAL
)
""")
conn.commit()

cur.execute("SELECT MAX(datetime) FROM weather_15min")
last_dt = cur.fetchone()[0]

today = datetime.today().date()

if last_dt is None:
    start_date = today - timedelta(days=7)
else:
    start_date = datetime.fromisoformat(last_dt).date() + timedelta(days=1)

current = start_date

while current <= today:
    date_str = current.strftime("%Y-%m-%d")
    print("Fetching weather data:", date_str)

    url = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast?"
        f"latitude={LAT}&longitude={LON}"
        f"&start_date={date_str}&end_date={date_str}"
        "&minutely_15=cloud_cover,temperature_2m,relative_humidity_2m,"
        "shortwave_radiation,direct_radiation,diffuse_radiation"
        "&timezone=Europe/London"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        times = data["minutely_15"]["time"]
        cloud = data["minutely_15"]["cloud_cover"]
        temp = data["minutely_15"]["temperature_2m"]
        humidity = data["minutely_15"]["relative_humidity_2m"]
        solar = data["minutely_15"]["shortwave_radiation"]
        direct = data["minutely_15"]["direct_radiation"]
        diffuse = data["minutely_15"]["diffuse_radiation"]

        for i in range(len(times)):
            cur.execute("""
            INSERT OR IGNORE INTO weather_15min
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                times[i],
                cloud[i],
                temp[i],
                humidity[i],
                solar[i],
                direct[i],
                diffuse[i]
            ))

        conn.commit()

    except Exception as e:
        print("Error retrieving weather data on", date_str, ":", e)

    current += timedelta(days=1)

conn.close()
print("Weather fetch finished.")