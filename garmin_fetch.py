from datetime import datetime, timedelta
import sqlite3
from garminconnect import Garmin
from config import GARMIN_EMAIL, GARMIN_PASSWORD, DB

conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS daily_metrics (
    date TEXT PRIMARY KEY,
    sleep_score INTEGER,
    total_sleep INTEGER,
    deep_sleep INTEGER,
    rem_sleep INTEGER,
    light_sleep INTEGER,
    awake_time INTEGER,
    sleep_start INTEGER,
    sleep_end INTEGER,
    avg_hr REAL,
    avg_resp REAL,
    resting_hr REAL,
    body_battery_change INTEGER,
    steps INTEGER,
    distance REAL,
    calories INTEGER,
    active_minutes INTEGER
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS sleep_stages (
    date TEXT,
    start_time TEXT,
    end_time TEXT,
    stage TEXT,
    duration_seconds INTEGER,
    PRIMARY KEY (date, start_time)
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS sleep_hr_timeseries (
    date TEXT,
    timestamp TEXT,
    hr_value INTEGER,
    PRIMARY KEY (date, timestamp)
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS sleep_spo2_timeseries (
    date TEXT,
    timestamp TEXT,
    spo2_value INTEGER,
    PRIMARY KEY (date, timestamp)
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS sleep_respiration_timeseries (
    date TEXT,
    timestamp TEXT,
    resp_value REAL,
    PRIMARY KEY (date, timestamp)
)
""")

conn.commit()

client = Garmin(GARMIN_EMAIL, GARMIN_PASSWORD)
client.login()

HARD_START = datetime(2026, 2, 16).date()

# For daily_metrics use the usual incremental logic
cur.execute("SELECT MAX(date) FROM daily_metrics")
last = cur.fetchone()[0]

if last is None:
    start_daily = HARD_START
else:
    start_daily = max(HARD_START, datetime.strptime(last, "%Y-%m-%d").date() + timedelta(days=1))

# For timeseries tables always backfill from HARD_START
cur.execute("SELECT MAX(date) FROM sleep_stages")
last_stages = cur.fetchone()[0]

if last_stages is None:
    start_timeseries = HARD_START
else:
    start_timeseries = max(HARD_START, datetime.strptime(last_stages, "%Y-%m-%d").date() + timedelta(days=1))

# Use the earlier of the two as overall start
start_date = min(start_daily, start_timeseries)

today = datetime.today().date()
current = start_date

while current <= today:
    date_str = current.strftime("%Y-%m-%d")
    print(f"fetching {date_str}")

    try:
        sleep_data = client.get_sleep_data(date_str)
        stats_data = client.get_stats(date_str)

        # daily summary
        if "dailySleepDTO" in sleep_data:
            dto = sleep_data["dailySleepDTO"]
            score = None
            if "sleepScores" in dto:
                score = dto["sleepScores"].get("overall", {}).get("value")

            cur.execute("""
            INSERT OR IGNORE INTO daily_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                date_str, score,
                dto.get("sleepTimeSeconds"), dto.get("deepSleepSeconds"),
                dto.get("remSleepSeconds"), dto.get("lightSleepSeconds"),
                dto.get("awakeSleepSeconds"), dto.get("sleepStartTimestampLocal"),
                dto.get("sleepEndTimestampLocal"), dto.get("avgHeartRate"),
                dto.get("averageRespirationValue"), sleep_data.get("restingHeartRate"),
                sleep_data.get("bodyBatteryChange"), stats_data.get("totalSteps"),
                stats_data.get("totalDistanceMeters"), stats_data.get("totalKilocalories"),
                stats_data.get("activeMinutes"),
            ))

        # sleep stages
        stages_list = sleep_data.get("sleepLevels") or sleep_data.get("sleepMovement") or []
        for level in stages_list:
            start_t = level.get("startGMT") or level.get("startTimeGMT")
            end_t   = level.get("endGMT")   or level.get("endTimeGMT")
            stage   = level.get("activityLevel") or level.get("sleepLevel")
            if not start_t:
                continue
            try:
                duration = int((datetime.fromisoformat(str(end_t)) - datetime.fromisoformat(str(start_t))).total_seconds())
            except Exception:
                duration = None
            cur.execute("INSERT OR IGNORE INTO sleep_stages VALUES (?,?,?,?,?)",
                        (date_str, str(start_t), str(end_t), str(stage), duration))

        # heart rate
        hr_list = sleep_data.get("sleepHeartRate") or sleep_data.get("heartRateValues") or []
        for entry in hr_list:
            if isinstance(entry, list) and len(entry) == 2:
                ts = datetime.utcfromtimestamp(entry[0]/1000).strftime("%Y-%m-%d %H:%M:%S")
                val = entry[1]
            elif isinstance(entry, dict):
                ts  = entry.get("timestamp") or entry.get("startGMT")
                val = entry.get("value") or entry.get("heartRate")
            else:
                continue
            if ts and val:
                cur.execute("INSERT OR IGNORE INTO sleep_hr_timeseries VALUES (?,?,?)",
                            (date_str, str(ts), val))

        # respiration
        for entry in sleep_data.get("wellnessEpochRespirationDataDTOList") or []:
            ts  = entry.get("startTimeGMT")
            val = entry.get("respirationValue")
            if ts and val:
                cur.execute("INSERT OR IGNORE INTO sleep_respiration_timeseries VALUES (?,?,?)",
                            (date_str, str(ts), val))

        # spo2
        for entry in sleep_data.get("wellnessEpochSPO2DataDTOList") or []:
            ts  = entry.get("startTimeGMT")
            val = entry.get("spo2Reading")
            if ts and val:
                cur.execute("INSERT OR IGNORE INTO sleep_spo2_timeseries VALUES (?,?,?)",
                            (date_str, str(ts), val))

        conn.commit()

    except Exception as e:
        print(f"error on {date_str}: {e}")

    current += timedelta(days=1)

conn.close()

conn = sqlite3.connect(DB)
cur = conn.cursor()
for table in ["daily_metrics", "sleep_stages", "sleep_hr_timeseries", "sleep_respiration_timeseries", "sleep_spo2_timeseries"]:
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    print(f"{table}: {cur.fetchone()[0]} rows")
conn.close()