# -*- coding: utf-8 -*-
import sqlite3, pandas as pd
from config import DB
conn = sqlite3.connect(DB)
df = pd.read_sql('SELECT * FROM weather_15min', conn)
conn.close()
day = df[df['datetime'].str.startswith('2026-02-23')]
day = day[day['datetime'].str[11:13].astype(int).between(8,18)]
print(day[['datetime','solar_radiation','direct_radiation','cloud_cover']].to_string())