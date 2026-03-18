import pandas as pd
import numpy as np
from scipy import stats
import datetime

INDOOR_FILE  = 'indoor_RAW_DATA.xlsx'
WEATHER_FILE = 'weather_15min_NEW.xlsx'
SLEEP_FILE   = 'sleep_summary_NEW.xlsx'
STAGES_FILE  = 'sleep_stages_NEW.csv'
HR_FILE      = 'sleep_hr_timeseries_NEW.csv'
RESP_FILE    = 'sleep_respiration_timeseries_NEW.csv'

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