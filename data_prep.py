import pandas as pd
import numpy as np
from scipy import stats
import datetime

INDOOR_FILE  = 'indoor_data.xlsx'
WEATHER_FILE = 'weather_15min.xlsx'
SLEEP_FILE   = 'sleep_summary.xlsx'
STAGES_FILE  = 'sleep_stages.csv'
HR_FILE      = 'sleep_hr_timeseries.csv'
RESP_FILE    = 'sleep_respiration_timeseries.csv'

DAYTIME_START  = 8
DAYTIME_END    = 18
PRESLEEP_START = 20
PRESLEEP_END   = 23

indoor = pd.read_excel(INDOOR_FILE, sheet_name='indoor')
indoor = indoor.drop(columns=['Unnamed: 4'], errors='ignore')
indoor['timestamp'] = pd.to_datetime(indoor['timestamp'])
indoor['brightness_pct'] = (4095 - indoor['light']) / 4095 * 100

# resample to 15-min grid and backfill missing values
# missing data concentrated in early morning (Feb 24 08:00-10:00)
indoor = (
    indoor
    .set_index('timestamp')
    .resample('15min')[['temperature', 'humidity', 'brightness_pct']]
    .mean()
    .bfill()
    .reset_index()
)
indoor['date'] = indoor['timestamp'].dt.date.astype(str)
indoor['hour'] = indoor['timestamp'].dt.hour

indoor_day = indoor[(indoor['hour'] >= DAYTIME_START) & (indoor['hour'] < DAYTIME_END)]
indoor_presleep = indoor[(indoor['hour'] >= PRESLEEP_START) & (indoor['hour'] < PRESLEEP_END)]

indoor_daily = indoor_day.groupby('date').agg(
    avg_brightness=('brightness_pct', 'mean'),
    avg_temp=('temperature', 'mean'),
    avg_humidity_day=('humidity', 'mean'),
).reset_index()

presleep_daily = indoor_presleep.groupby('date').agg(
    avg_humidity_presleep=('humidity', 'mean'),
    avg_temp_presleep=('temperature', 'mean'),
).reset_index()

weather = pd.read_excel(WEATHER_FILE, sheet_name='Sheet1')
weather['datetime'] = pd.to_datetime(weather['datetime'])
weather['date'] = weather['datetime'].dt.date.astype(str)
weather['hour'] = weather['datetime'].dt.hour

weather_day = weather[(weather['hour'] >= DAYTIME_START) & (weather['hour'] < DAYTIME_END)]
weather_daily = weather_day.groupby('date').agg(
    avg_solar=('solar_radiation', 'mean'),
    max_solar=('solar_radiation', 'max'),
    avg_cloud=('cloud_cover', 'mean'),
    outdoor_temp=('temperature', 'mean'),
    outdoor_humidity=('humidity', 'mean'),
).reset_index()

sleep = pd.read_excel(SLEEP_FILE, sheet_name='Sheet1')
sleep['date'] = pd.to_datetime(sleep['date']).dt.date
sleep['date_str'] = sleep['date'].astype(str)
sleep['sleep_start_dt'] = pd.to_datetime(sleep['sleep_start'], unit='ms')
sleep['sleep_end_dt'] = pd.to_datetime(sleep['sleep_end'], unit='ms')
# Garmin labels sleep as D+1, so preceding daytime is D
sleep['light_date'] = sleep['date'].apply(lambda d: str(d - datetime.timedelta(days=1)))
sleep['total_sleep_h'] = sleep['total_sleep'] / 3600
sleep['deep_pct'] = sleep['deep_sleep'] / sleep['total_sleep'] * 100
sleep['rem_pct'] = sleep['rem_sleep'] / sleep['total_sleep'] * 100
sleep['light_pct'] = sleep['light_sleep'] / sleep['total_sleep'] * 100
sleep['awake_h'] = sleep['awake_time'] / 3600

stages = pd.read_csv(STAGES_FILE)
stages['start_dt'] = pd.to_datetime(stages['start_time'])
stages['end_dt'] = pd.to_datetime(stages['end_time'])
stages['stage_name'] = stages['stage'].map(
    {1.0: 'light', 2.0: 'rem', 3.0: 'deep', -1.0: 'awake'}
).fillna('awake')

hr = pd.read_csv(HR_FILE)
resp = pd.read_csv(RESP_FILE)
hr['ts'] = pd.to_datetime(hr['timestamp'], unit='ms')
resp['ts'] = pd.to_datetime(resp['timestamp'], unit='ms')

hr_stats = hr.groupby('date')['hr_value'].agg(
    hr_mean='mean', hr_min='min', hr_max='max', hr_std='std'
).reset_index().rename(columns={'date': 'date_str'})

resp_stats = resp.groupby('date')['resp_value'].agg(
    resp_mean='mean', resp_std='std'
).reset_index().rename(columns={'date': 'date_str'})

def get_stage_features(df):
    t0 = df['start_dt'].min()
    deep = df[df['stage'] == 3.0].sort_values('start_dt')
    rem = df[df['stage'] == 2.0].sort_values('start_dt')
    awake = df[df['stage'].isna()]
    seq = df.sort_values('start_dt')['stage'].tolist()
    n_trans = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
    return pd.Series({
        'total_awake_min': awake['duration_seconds'].sum() / 60,
        'first_deep_min': (deep.iloc[0]['start_dt'] - t0).total_seconds() / 60 if len(deep) > 0 else np.nan,
        'first_rem_min': (rem.iloc[0]['start_dt'] - t0).total_seconds() / 60 if len(rem) > 0 else np.nan,
        'n_transitions': n_trans,
    })

stage_feats = stages.groupby('date').apply(get_stage_features).reset_index()
stage_feats = stage_feats.rename(columns={'date': 'date_str'})

sleep_hum = []
for _, row in sleep.iterrows():
    mask = (indoor['timestamp'] >= row['sleep_start_dt']) & (indoor['timestamp'] <= row['sleep_end_dt'])
    vals = indoor.loc[mask, 'humidity']
    sleep_hum.append({
        'date_str': row['date_str'],
        'avg_humidity_sleep': vals.mean() if len(vals) >= 3 else np.nan,
    })
sleep_hum = pd.DataFrame(sleep_hum)

env = indoor_daily.merge(presleep_daily, on='date', how='left')
env = env.merge(weather_daily, on='date', how='left')
env = env.rename(columns={'date': 'light_date'})

merged = env.merge(sleep, on='light_date', how='inner')
merged = merged.merge(sleep_hum, on='date_str', how='left')
merged = merged.merge(hr_stats, on='date_str', how='left')
merged = merged.merge(resp_stats, on='date_str', how='left')
merged = merged.merge(stage_feats, on='date_str', how='left')

# 15-min timeseries aligned on timestamp for solar vs brightness
indoor['timestamp_r'] = indoor['timestamp'].dt.round('15min')
indoor_15 = indoor[
    (indoor['hour'] >= DAYTIME_START) & (indoor['hour'] < DAYTIME_END)
][['timestamp_r', 'timestamp', 'brightness_pct']].copy()

weather_15 = weather[
    (weather['hour'] >= DAYTIME_START) & (weather['hour'] < DAYTIME_END)
][['datetime', 'solar_radiation', 'cloud_cover']].copy()

ts_aligned = indoor_15.merge(
    weather_15, left_on='timestamp_r', right_on='datetime', how='inner'
).dropna()

# ── data volume summary ───────────────────────────────────────────────────────
if __name__ == '__main__':
    # raw counts (before resample/processing)
    indoor_raw = pd.read_excel(INDOOR_FILE, sheet_name='indoor')
    indoor_raw = indoor_raw.drop(columns=['Unnamed: 4'], errors='ignore')
    indoor_raw['timestamp'] = pd.to_datetime(indoor_raw['timestamp'])

    weather_raw = pd.read_excel(WEATHER_FILE, sheet_name='Sheet1')
    weather_raw['datetime'] = pd.to_datetime(weather_raw['datetime'])
    w_filtered = weather_raw[
        (weather_raw['datetime'].dt.date >= pd.Timestamp('2026-02-16').date()) &
        (weather_raw['datetime'].dt.date <= pd.Timestamp('2026-02-28').date())
    ]

    sleep_raw2 = pd.read_excel(SLEEP_FILE, sheet_name='Sheet1')
    sleep_raw2['date_dt'] = pd.to_datetime(sleep_raw2['date'])
    s_filtered = sleep_raw2[
        (sleep_raw2['date_dt'].dt.date >= pd.Timestamp('2026-02-16').date()) &
        (sleep_raw2['date_dt'].dt.date <= pd.Timestamp('2026-03-01').date())
    ]

    hr_raw2     = pd.read_csv(HR_FILE)
    resp_raw2   = pd.read_csv(RESP_FILE)
    stages_raw2 = pd.read_csv(STAGES_FILE)
    hr_f   = hr_raw2[(hr_raw2['date'] >= '2026-02-16') & (hr_raw2['date'] <= '2026-03-01')]
    resp_f = resp_raw2[(resp_raw2['date'] >= '2026-02-16') & (resp_raw2['date'] <= '2026-03-01')]
    st_f   = stages_raw2[(stages_raw2['date'] >= '2026-02-16') & (stages_raw2['date'] <= '2026-03-01')]

    print("\n=== RAW DATA (before processing) ===")
    print(f"Indoor sensor:      {len(indoor_raw)} records (raw, before resample)")
    print(f"Outdoor weather:    {len(w_filtered)} records, Feb 16-28")
    print(f"Garmin daily:       {len(s_filtered)} nights, Feb 16 – Mar 1")
    print(f"Garmin HR:          {len(hr_f)} records, {hr_f['date'].nunique()} nights")
    print(f"Garmin respiration: {len(resp_f)} records, {resp_f['date'].nunique()} nights")
    print(f"Garmin stages:      {len(st_f)} segments, {st_f['date'].nunique()} nights")

    print("\n=== PROCESSED DATA (after resample/cleaning) ===")
    print(f"Indoor sensor:      {len(indoor)} records (after 15-min resample + backfill)")
    print(f"Final dataset:      n = {len(merged)} days, {merged['light_date'].min()} to {merged['light_date'].max()}")
    print(f"15-min ts_aligned:  {len(ts_aligned)} points (daytime 08:00-18:00)")

    print("\n=== DAYTIME BRIGHTNESS MISSING DATA (08:00-18:00, expected 40 slots/day) ===")
    indoor_raw['ts_15'] = indoor_raw['timestamp'].dt.round('15min')
    indoor_raw['date_only'] = indoor_raw['ts_15'].dt.date
    indoor_raw['hour_only'] = indoor_raw['ts_15'].dt.hour
    daytime_raw = indoor_raw[(indoor_raw['hour_only'] >= 8) & (indoor_raw['hour_only'] < 18)]
    for date in sorted(daytime_raw['date_only'].unique()):
        day = daytime_raw[daytime_raw['date_only'] == date]
        actual = len(day)
        expected = 40
        missing = expected - actual
        pct = actual / expected * 100
        if missing > 0:
            expected_slots = pd.date_range(f'{date} 08:00', f'{date} 17:45', freq='15min')
            actual_slots = set(day['ts_15'].dt.floor('15min'))
            missing_slots = [str(s.time()) for s in expected_slots if s not in actual_slots]
            print(f"  {date}: {actual}/{expected} ({pct:.0f}%), missing: {missing_slots}")
        else:
            print(f"  {date}: {actual}/{expected} (100%)")

    print("\n=== SLEEP-PERIOD HUMIDITY & TEMPERATURE ===")
    sleep_raw3 = pd.read_excel(SLEEP_FILE, sheet_name='Sheet1')
    sleep_raw3['sleep_start_dt'] = pd.to_datetime(sleep_raw3['sleep_start'], unit='ms')
    sleep_raw3['sleep_end_dt']   = pd.to_datetime(sleep_raw3['sleep_end'],   unit='ms')
    sleep_raw3['date_str'] = pd.to_datetime(sleep_raw3['date']).dt.date.astype(str)
    sleep_raw3 = sleep_raw3[
        (sleep_raw3['date_str'] >= '2026-02-16') &
        (sleep_raw3['date_str'] <= '2026-03-01')
    ]
    for _, row in sleep_raw3.iterrows():
        mask = (indoor_raw['timestamp'] >= row['sleep_start_dt']) & (indoor_raw['timestamp'] <= row['sleep_end_dt'])
        vals = indoor_raw[mask]
        duration_h = (row['sleep_end_dt'] - row['sleep_start_dt']).total_seconds() / 3600
        expected = int(duration_h * 4)
        actual = len(vals)
        pct = actual / expected * 100 if expected > 0 else 0
        hum_mean = f"{vals['humidity'].mean():.1f}%" if actual >= 3 else "NaN"
        temp_mean = f"{vals['temperature'].mean():.1f}°C" if actual >= 3 else "NaN"
        status = "(INSUFFICIENT)" if actual < 3 else ""
        print(f"  {row['date_str']}: {actual}/{expected} ({pct:.0f}%) {status}")
        print(f"    humidity={hum_mean}, temperature={temp_mean}, sleep {row['sleep_start_dt'].strftime('%H:%M')}–{row['sleep_end_dt'].strftime('%H:%M')}")