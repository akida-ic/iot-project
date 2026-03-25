import sys, os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data_prep import merged, ts_aligned

os.makedirs('figures', exist_ok=True)
merged['date_dt'] = pd.to_datetime(merged['light_date'])

SOLAR     = '#f59e0b'
BRIGHT    = '#a78bfa'
DEEP      = '#818cf8'
REM       = '#60a5fa'
LIGHT     = '#cbd5e1'
HUM_DAY   = '#38bdf8'
HUM_PRE   = '#818cf8'
HUM_SLEEP = '#60a5fa'
HR        = '#f87171'
CYAN      = '#06b6d4'


def corr(x, y, label=''):
    sub = merged[[x, y]].dropna()
    r, p = stats.pearsonr(sub[x], sub[y])
    sig = '**' if p < 0.05 else ('*' if p < 0.1 else 'ns')
    print(f"  [{sig}]  r={r:+.3f}  p={p:.3f}  n={len(sub)}  {label}")
    return r, p


# ── SECTION 1: Solar / brightness ────────────────────────────────────────────
print("=" * 60)
print("SECTION 1: Solar / cloud -> brightness")
print("=" * 60)
corr('avg_solar', 'avg_brightness', 'solar -> brightness')
corr('avg_cloud', 'avg_brightness', 'cloud -> brightness')
sub = merged[['avg_solar', 'avg_brightness']].dropna()
slope, intercept, r_ols, p_ols, _ = stats.linregress(sub['avg_solar'], sub['avg_brightness'])
print(f"  OLS: brightness = {slope:.3f} x solar + {intercept:.2f},  R2={r_ols**2:.3f}")
print(f"  Daily mean brightness: {merged['avg_brightness'].mean():.1f}%")
print(f"  Daily mean cloud cover: {merged['avg_cloud'].mean():.1f}%")

r_ts, p_ts = stats.pearsonr(ts_aligned['solar_radiation'], ts_aligned['brightness_pct'])
print(f"  15-min ts_aligned: r={r_ts:.3f}  p={p_ts:.4f}  n={len(ts_aligned)}")


# ── SECTION 2: Brightness -> sleep (lag 1d) ───────────────────────────────────
print()
print("=" * 60)
print("SECTION 2: Brightness -> sleep (lag 1d)")
print("=" * 60)
for tgt, lbl in [
    ('sleep_score',     'sleep score'),
    ('deep_pct',        'deep %'),
    ('rem_pct',         'REM %'),
    ('total_sleep_h',   'total sleep h'),
    ('hr_mean',         'mean HR'),
    ('resp_mean',       'mean resp'),
    ('resp_std',        'resp variability'),
    ('total_awake_min', 'awake min'),
    ('first_deep_min',  'time to first deep'),
    ('n_transitions',   'stage transitions'),
]:
    corr('avg_brightness', tgt, f'brightness -> {lbl}')

print()
print("Solar -> sleep")
for tgt, lbl in [
    ('sleep_score', 'sleep score'),
    ('deep_pct',    'deep %'),
    ('hr_mean',     'mean HR'),
]:
    corr('avg_solar', tgt, f'solar -> {lbl}')


# ── SECTION 3: Humidity -> sleep ──────────────────────────────────────────────
print()
print("=" * 60)
print("SECTION 3: Humidity -> sleep")
print("=" * 60)
for hum, hlbl in [
    ('avg_humidity_day',      'daytime (08-18)'),
    ('avg_humidity_presleep', 'pre-sleep (20-23)'),
    ('avg_humidity_sleep',    'sleep period'),
]:
    print(f"  {hlbl}")
    for tgt, lbl in [
        ('sleep_score', 'sleep score'),
        ('hr_mean',     'mean HR'),
        ('resp_mean',   'resp rate'),
        ('deep_pct',    'deep %'),
    ]:
        corr(hum, tgt, lbl)

# threshold analysis
print()
print("  Threshold analysis (sleep-period humidity):")
sub_hs = merged[['avg_humidity_sleep', 'sleep_score']].dropna()
print(f"  n valid nights = {len(sub_hs)}")
print(f"  humidity range: {sub_hs['avg_humidity_sleep'].min():.1f} - {sub_hs['avg_humidity_sleep'].max():.1f}%")
for threshold in [35]:
    high = sub_hs[sub_hs['avg_humidity_sleep'] >  threshold]['sleep_score'].mean()
    low  = sub_hs[sub_hs['avg_humidity_sleep'] <= threshold]['sleep_score'].mean()
    n_high = len(sub_hs[sub_hs['avg_humidity_sleep'] >  threshold])
    n_low  = len(sub_hs[sub_hs['avg_humidity_sleep'] <= threshold])
    print(f"  >{threshold}%: n={n_high}, avg sleep score={high:.1f}")
    print(f"  <={threshold}%: n={n_low},  avg sleep score={low:.1f}")


# ── SECTION 4: Temperature -> sleep ──────────────────────────────────────────
print()
print("=" * 60)
print("SECTION 4: Temperature -> sleep")
print("=" * 60)
for tgt, lbl in [
    ('sleep_score', 'sleep score'),
    ('deep_pct',    'deep %'),
    ('hr_mean',     'mean HR'),
]:
    corr('avg_temp', tgt, f'temp -> {lbl}')


# ── SECTION 5: Confounding ────────────────────────────────────────────────────
print()
print("=" * 60)
print("SECTION 5: Confounding variables")
print("=" * 60)
corr('outdoor_humidity', 'sleep_score',     'outdoor humidity -> sleep score')
corr('outdoor_humidity', 'avg_brightness',  'outdoor humidity -> brightness')
corr('outdoor_humidity', 'avg_humidity_day','outdoor humidity -> indoor humidity day')


# ── SECTION 6: Multiple regression (controlling for outdoor humidity) ─────────
print()
print("=" * 60)
print("SECTION 6: Multiple regression — controlling for outdoor humidity")
print("=" * 60)

try:
    import statsmodels.api as sm

    # Model 1: brightness -> deep_pct (univariate baseline)
    sub1 = merged[['avg_brightness', 'deep_pct']].dropna()
    X1 = sm.add_constant(sub1[['avg_brightness']])
    m1 = sm.OLS(sub1['deep_pct'], X1).fit()
    print(f"\n  Model 1 (univariate): brightness -> deep_pct")
    print(f"  coef brightness={m1.params['avg_brightness']:.4f}  "
          f"p={m1.pvalues['avg_brightness']:.3f}  R2={m1.rsquared:.3f}")

    # Model 2: brightness + outdoor_humidity -> deep_pct
    sub2 = merged[['avg_brightness', 'outdoor_humidity', 'deep_pct']].dropna()
    X2 = sm.add_constant(sub2[['avg_brightness', 'outdoor_humidity']])
    m2 = sm.OLS(sub2['deep_pct'], X2).fit()
    print(f"\n  Model 2 (multivariate): brightness + outdoor_humidity -> deep_pct")
    print(f"  coef brightness={m2.params['avg_brightness']:.4f}  "
          f"p={m2.pvalues['avg_brightness']:.3f}")
    print(f"  coef outdoor_humidity={m2.params['outdoor_humidity']:.4f}  "
          f"p={m2.pvalues['outdoor_humidity']:.3f}")
    print(f"  R2={m2.rsquared:.3f}  adj R2={m2.rsquared_adj:.3f}  n={len(sub2)}")

    # Model 3: brightness + outdoor_humidity -> sleep_score
    sub3 = merged[['avg_brightness', 'outdoor_humidity', 'sleep_score']].dropna()
    X3 = sm.add_constant(sub3[['avg_brightness', 'outdoor_humidity']])
    m3 = sm.OLS(sub3['sleep_score'], X3).fit()
    print(f"\n  Model 3 (multivariate): brightness + outdoor_humidity -> sleep_score")
    print(f"  coef brightness={m3.params['avg_brightness']:.4f}  "
          f"p={m3.pvalues['avg_brightness']:.3f}")
    print(f"  coef outdoor_humidity={m3.params['outdoor_humidity']:.4f}  "
          f"p={m3.pvalues['outdoor_humidity']:.3f}")
    print(f"  R2={m3.rsquared:.3f}  adj R2={m3.rsquared_adj:.3f}  n={len(sub3)}")

except ImportError:
    print("  statsmodels not installed — run: pip install statsmodels")


# ── SECTION 7: Moving average & anomaly detection ────────────────────────────
print()
print("=" * 60)
print("SECTION 7: Moving average & anomaly detection")
print("=" * 60)

# brightness anomalies
ts_b   = ts_aligned.set_index('timestamp')['brightness_pct']
ma_b   = ts_b.rolling(window=8, center=True).mean()
std_b  = (ts_b - ma_b).std()
upper_b = ma_b + 1.5 * std_b
lower_b = ma_b - 1.5 * std_b
anom_b  = ts_b[(ts_b > upper_b) | (ts_b < lower_b)]
print(f"  Brightness anomalies (±1.5 SD, 2h MA): n={len(anom_b)}")

# HR anomalies
hr_raw = pd.read_csv('sleep_hr_timeseries.csv')
hr_raw['ts'] = pd.to_datetime(hr_raw['timestamp'], unit='ms')
hr_valid = hr_raw[hr_raw['date'] <= '2026-02-28'].copy()
hr_daily = hr_valid.groupby('date')['hr_value'].mean().reset_index()
hr_daily.columns = ['date_str', 'hr_mean_night']
hr_daily = hr_daily.merge(merged[['date_str', 'sleep_score']], on='date_str', how='left')
mean_hr   = hr_daily['hr_mean_night'].mean()
std_hr    = hr_daily['hr_mean_night'].std()
upper_hr  = mean_hr + 1.5 * std_hr
lower_hr  = mean_hr - 1.5 * std_hr
anomalies = hr_daily[(hr_daily['hr_mean_night'] > upper_hr) | (hr_daily['hr_mean_night'] < lower_hr)]
print(f"  HR overall mean={mean_hr:.1f} bpm, SD={std_hr:.1f}")
print(f"  HR anomaly threshold: >{upper_hr:.1f} or <{lower_hr:.1f}")
print(f"  HR anomalies:")
print(anomalies[['date_str', 'hr_mean_night', 'sleep_score']].to_string(index=False))


# ── SECTION 7: Within-night humidity correlations ────────────────────────────
print()
print("=" * 60)
print("SECTION 8: Within-night humidity correlations")
print("=" * 60)

indoor_raw = pd.read_excel('indoor_data.xlsx', sheet_name='indoor')
indoor_raw = indoor_raw.drop(columns=['Unnamed: 4'], errors='ignore')
indoor_raw['timestamp'] = pd.to_datetime(indoor_raw['timestamp'])

hr_ts   = pd.read_csv('sleep_hr_timeseries.csv')
resp_ts = pd.read_csv('sleep_respiration_timeseries.csv')
hr_ts['ts']   = pd.to_datetime(hr_ts['timestamp'], unit='ms')
resp_ts['ts'] = pd.to_datetime(resp_ts['timestamp'], unit='ms')

sleep_raw = pd.read_excel('sleep_summary.xlsx', sheet_name='Sheet1')
sleep_raw['sleep_start_dt'] = pd.to_datetime(sleep_raw['sleep_start'], unit='ms')
sleep_raw['sleep_end_dt']   = pd.to_datetime(sleep_raw['sleep_end'],   unit='ms')
sleep_raw['date_str'] = pd.to_datetime(sleep_raw['date']).dt.date.astype(str)
valid_nights = sleep_raw[
    (sleep_raw['sleep_start_dt'] >= pd.Timestamp('2026-02-17')) &
    (sleep_raw['sleep_end_dt']   <  pd.Timestamp('2026-03-01'))
].copy()

r_hr_list, r_resp_list, hum_ranges = [], [], []
for _, night in valid_nights.iterrows():
    mask_in   = (indoor_raw['timestamp'] >= night['sleep_start_dt']) & \
                (indoor_raw['timestamp'] <= night['sleep_end_dt'])
    mask_hr   = hr_ts['ts'].between(night['sleep_start_dt'], night['sleep_end_dt'])
    mask_resp = resp_ts['ts'].between(night['sleep_start_dt'], night['sleep_end_dt'])

    indoor_n = indoor_raw[mask_in][['timestamp', 'humidity']].copy()
    indoor_n['ts_15'] = indoor_n['timestamp'].dt.round('15min')
    indoor_n = indoor_n.groupby('ts_15')['humidity'].mean().reset_index()

    hr_n = hr_ts[mask_hr].copy()
    hr_n['ts_15'] = hr_n['ts'].dt.round('15min')
    hr_15 = hr_n.groupby('ts_15')['hr_value'].mean().reset_index()

    resp_n = resp_ts[mask_resp].copy()
    resp_n['ts_15'] = resp_n['ts'].dt.round('15min')
    resp_15 = resp_n.groupby('ts_15')['resp_value'].mean().reset_index()

    m = indoor_n.merge(hr_15, on='ts_15').merge(resp_15, on='ts_15')
    if len(m) < 5:
        continue

    r_hr,   _ = stats.pearsonr(m['humidity'], m['hr_value'])
    r_resp, _ = stats.pearsonr(m['humidity'], m['resp_value'])
    hum_range  = m['humidity'].max() - m['humidity'].min()

    r_hr_list.append(r_hr)
    r_resp_list.append(r_resp)
    hum_ranges.append(hum_range)
    print(f"  {night['date_str']}: r_hr={r_hr:+.3f}  r_resp={r_resp:+.3f}  hum_range={hum_range:.1f}%")

print(f"  Mean r_hr={np.mean(r_hr_list):+.3f}  range={min(r_hr_list):.2f} to {max(r_hr_list):.2f}")
print(f"  Mean r_resp={np.mean(r_resp_list):+.3f}  range={min(r_resp_list):.2f} to {max(r_resp_list):.2f}")
print(f"  Mean within-night hum range={np.mean(hum_ranges):.1f}%  "
      f"min={min(hum_ranges):.1f}%  max={max(hum_ranges):.1f}%")


# ── FIGURES ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 5))
ax2 = ax.twinx()
ax.fill_between(ts_aligned['timestamp'], ts_aligned['solar_radiation'],
                color=SOLAR, alpha=0.3, label='Solar radiation (W/m²)')
ax.plot(ts_aligned['timestamp'], ts_aligned['solar_radiation'],
        color=SOLAR, linewidth=0.8, alpha=0.9)
ax2.fill_between(ts_aligned['timestamp'], ts_aligned['brightness_pct'],
                 color=BRIGHT, alpha=0.3, label='Indoor brightness (%)')
ax2.plot(ts_aligned['timestamp'], ts_aligned['brightness_pct'],
         color=BRIGHT, linewidth=0.8, alpha=0.9)
ax.set_ylabel('Solar radiation (W/m²)', color=SOLAR, fontsize=10)
ax2.set_ylabel('Indoor brightness (%)', color=BRIGHT, fontsize=10)
ax.set_ylim(0, 550)
ax2.set_ylim(0, 105)
ax.set_title(
    f'Outdoor Solar Radiation vs Indoor Brightness — 15-min Time Series\n'
    f'London, Feb 2026  (08:00–18:00)  |  r = {r_ts:.3f}, p < 0.001, n = {len(ts_aligned)}',
    fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.2)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig0_solar_vs_brightness_15min.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nsaved fig0_solar_vs_brightness_15min")


fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Indoor Daylighting & Sleep — Daily Overview\nLondon, Feb 2026',
             fontsize=13, fontweight='bold', y=0.98)

ax = axes[0]
axb = ax.twinx()
ax.bar(merged['date_dt'], merged['avg_solar'], color=SOLAR, alpha=0.55, width=0.7,
       label='Solar (W/m²)')
axb.plot(merged['date_dt'], merged['avg_brightness'], color=BRIGHT, marker='o',
         linewidth=2, markersize=5, label='Brightness (%)')
ax.set_ylabel('Solar radiation (W/m²)', color=SOLAR, fontsize=10)
axb.set_ylabel('Indoor brightness (%)', color=BRIGHT, fontsize=10)
ax.set_ylim(0, 400)
axb.set_ylim(0, 80)
lines = ax.get_legend_handles_labels()[0] + axb.get_legend_handles_labels()[0]
labels = ax.get_legend_handles_labels()[1] + axb.get_legend_handles_labels()[1]
ax.legend(lines, labels, loc='upper right', fontsize=9)
ax.set_title('Panel A — Daily mean solar vs indoor brightness (08:00–18:00)', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.text(0.01, 0.88, f'r = +{r_ols:.3f}, p < 0.001',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffbe6', alpha=0.8))

ax = axes[1]
score_colors = ['#34d399' if s >= 80 else '#fbbf24' if s >= 65 else '#f87171'
                for s in merged['sleep_score']]
ax.bar(merged['date_dt'], merged['sleep_score'], color=score_colors, alpha=0.75, width=0.7)
ax.axhline(80, color='#34d399', linewidth=1, linestyle='--', alpha=0.5)
ax.axhline(65, color='#fbbf24', linewidth=1, linestyle='--', alpha=0.5)
axb = ax.twinx()
axb.plot(merged['date_dt'], merged['avg_brightness'], color=BRIGHT,
         linewidth=1.5, linestyle=':', alpha=0.6)
ax.set_ylabel('Sleep score', fontsize=10)
axb.set_ylabel('Indoor brightness (%)', color=BRIGHT, fontsize=10)
ax.set_ylim(40, 100)
axb.set_ylim(0, 80)
ax.set_title('Panel B — Sleep score (green ≥80, yellow 65–79, red <65)', fontsize=10)
ax.grid(axis='y', alpha=0.3)

ax = axes[2]
ax.stackplot(
    merged['date_dt'],
    merged['deep_pct'],
    merged['rem_pct'],
    100 - merged['deep_pct'] - merged['rem_pct'] - merged['awake_h'] / merged['total_sleep_h'] * 100,
    labels=['Deep', 'REM', 'Light'],
    colors=[DEEP, REM, LIGHT], alpha=0.85)
ax.set_ylabel('Sleep stage composition (%)', fontsize=10)
ax.set_title('Panel C — Sleep stage composition', fontsize=10)
ax.set_ylim(0, 100)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
plt.tight_layout()
fig.text(0.5, -0.01,
         'London, Feb 16–28 2026  ·  ESP32 LDR sensor (08:00–18:00)  ·  Open-Meteo API  ·  Garmin Venu',
         ha='center', fontsize=8, color='#666', style='italic')
plt.savefig('figures/fig1_daily_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved fig1_daily_overview")


def scatter_fit(ax, x, y, xlabel, ylabel, title, color):
    sub = merged[[x, y]].dropna()
    ax.scatter(sub[x], sub[y], color=color, s=55, alpha=0.85,
               edgecolors='white', linewidth=0.5, zorder=3)
    m, b, r, p, _ = stats.linregress(sub[x], sub[y])
    xline = np.linspace(sub[x].min(), sub[x].max(), 100)
    ax.plot(xline, m * xline + b, color=color, linewidth=2, alpha=0.9)
    sig = '**' if p < 0.05 else ('*' if p < 0.1 else '')
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f'{title}\nr={r:.3f}{sig}, p={p:.3f}, n={len(sub)}', fontsize=9)
    ax.grid(alpha=0.25)
    for _, row in sub.iterrows():
        ax.annotate(merged.loc[row.name, 'light_date'][5:], (row[x], row[y]),
                    textcoords='offset points', xytext=(4, 3), fontsize=6, color='#888')

# fig2a — daylighting correlations (top row)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle('Figure 1. Correlation Analysis — Daylighting & Sleep Quality',
             fontsize=13, fontweight='bold')
scatter_fit(axes[0], 'avg_solar',       'avg_brightness', 'Solar radiation (W/m²)',  'Indoor brightness (%)',       'A  Solar -> Brightness',      SOLAR)
scatter_fit(axes[1], 'avg_brightness',  'deep_pct',       'Indoor brightness (%)',   'Deep sleep (%)',              'B  Brightness -> Deep sleep', DEEP)
scatter_fit(axes[2], 'avg_brightness',  'resp_std',       'Indoor brightness (%)',   'Respiration variability (SD)','C  Brightness -> Resp var',   CYAN)
plt.tight_layout()
fig.text(0.5, -0.04,
         '** p < 0.05  ·  * p < 0.1  ·  ns p ≥ 0.1  ·  n = 13 days (daylighting, lag=1d)',
         ha='center', fontsize=8, color='#666', style='italic')
plt.savefig('figures/fig2a_daylighting_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved fig2a_daylighting_scatter")

# fig2b — humidity correlations (bottom row)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle('Figure 2. Correlation Analysis — Sleep-period Humidity & Sleep Quality',
             fontsize=13, fontweight='bold')
scatter_fit(axes[0], 'avg_humidity_sleep', 'sleep_score', 'Sleep-period humidity (%)', 'Sleep score',          'D  Sleep humidity -> Sleep score', HUM_SLEEP)
scatter_fit(axes[1], 'avg_humidity_sleep', 'hr_mean',     'Sleep-period humidity (%)', 'Mean sleep HR (bpm)',  'E  Sleep humidity -> Sleep HR',    HR)
scatter_fit(axes[2], 'avg_solar',          'deep_pct',    'Solar radiation (W/m²)',    'Deep sleep (%)',       'F  Solar -> Deep sleep',           SOLAR)
plt.tight_layout()
fig.text(0.5, -0.04,
         '** p < 0.05  ·  * p < 0.1  ·  ns p ≥ 0.1  ·  n = 12 nights (sleep-period humidity)  ·  n = 13 days (solar)',
         ha='center', fontsize=8, color='#666', style='italic')
plt.savefig('figures/fig2b_humidity_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved fig2b_humidity_scatter")


fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle('Humidity Window Comparison', fontsize=12, fontweight='bold')
for ax, (hum, hlbl, color) in zip(axes, [
    ('avg_humidity_day',      'Daytime (08:00-18:00)',   HUM_DAY),
    ('avg_humidity_presleep', 'Pre-sleep (20:00-23:00)', HUM_PRE),
    ('avg_humidity_sleep',    'Sleep period',            HUM_SLEEP),
]):
    sub = merged[[hum, 'sleep_score']].dropna()
    ax.scatter(sub[hum], sub['sleep_score'], color=color, s=55, alpha=0.85,
               edgecolors='white', linewidth=0.5)
    if len(sub) >= 5:
        m, b, r, p, _ = stats.linregress(sub[hum], sub['sleep_score'])
        xline = np.linspace(sub[hum].min(), sub[hum].max(), 100)
        ax.plot(xline, m * xline + b, color=color, linewidth=2)
        sig = '**' if p < 0.05 else ('*' if p < 0.1 else 'ns')
        ax.set_title(f'{hlbl}\nr={r:.3f}{sig}, p={p:.3f}', fontsize=9)
    ax.set_xlabel('Humidity (%)', fontsize=9)
    ax.set_ylabel('Sleep score', fontsize=9)
    ax.grid(alpha=0.25)
plt.tight_layout()
fig.text(0.5, -0.04,
         '** p < 0.05  ·  * p < 0.1  ·  ns p ≥ 0.1  ·  n = 13 days (daytime/pre-sleep)  ·  n = 12 nights (sleep-period)',
         ha='center', fontsize=8, color='#666', style='italic')
plt.savefig('figures/fig3_humidity_windows.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved fig3_humidity_windows")


predictors  = ['avg_brightness', 'avg_solar', 'avg_humidity_sleep', 'avg_temp']
outcomes    = ['sleep_score', 'deep_pct', 'rem_pct', 'hr_mean', 'resp_mean', 'resp_std']
pred_labels = ['Indoor\nbrightness', 'Outdoor\nsolar', 'Sleep-period\nhumidity', 'Indoor\ntemp']
out_labels  = ['Sleep\nscore', 'Deep\nsleep %', 'REM %', 'Sleep\nHR', 'Resp\nmean', 'Resp\nvariability']

matrix  = np.zeros((len(predictors), len(outcomes)))
pmatrix = np.ones((len(predictors), len(outcomes)))
for i, pred in enumerate(predictors):
    for j, out in enumerate(outcomes):
        sub = merged[[pred, out]].dropna()
        if len(sub) >= 5:
            r, p = stats.pearsonr(sub[pred], sub[out])
            matrix[i, j]  = r
            pmatrix[i, j] = p

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle('Figure 4. Pearson r — Environmental predictors vs sleep outcomes (lag=1d)',
             fontsize=11, fontweight='bold')
im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='Pearson r')
ax.set_xticks(range(len(outcomes)))
ax.set_yticks(range(len(predictors)))
ax.set_xticklabels(out_labels, fontsize=9)
ax.set_yticklabels(pred_labels, fontsize=9)
for i in range(len(predictors)):
    for j in range(len(outcomes)):
        r, p = matrix[i, j], pmatrix[i, j]
        sig = '**' if p < 0.05 else ('*' if p < 0.1 else '')
        ax.text(j, i, f'{r:.2f}{sig}', ha='center', va='center', fontsize=9,
                color='white' if abs(r) > 0.5 else '#444',
                fontweight='bold' if sig else 'normal')
ax.set_title('** p<0.05   * p<0.1', fontsize=9, pad=8)
plt.tight_layout()
fig.text(0.5, -0.04,
         '** p < 0.05  ·  * p < 0.1  ·  n = 13 days (daylighting, lag=1d)  ·  n = 12 nights (sleep-period humidity)',
         ha='center', fontsize=8, color='#666', style='italic')
plt.savefig('figures/fig4_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved fig4_correlation_heatmap")


fig, axes = plt.subplots(2, 1, figsize=(14, 9))
fig.suptitle('Figure 3. Moving Average & Anomaly Detection', fontsize=13, fontweight='bold')

ax = axes[0]
ax.plot(ts_b.index, ts_b.values, color='#a78bfa', linewidth=0.7, alpha=0.6,
        label='Brightness (%)')
ax.plot(ma_b.index, ma_b.values, color='#7c3aed', linewidth=2,
        label='2h Moving average')
ax.fill_between(ma_b.index, lower_b.values, upper_b.values,
                color='#a78bfa', alpha=0.12, label='±1.5 SD band')
ax.scatter(anom_b.index, anom_b.values, color='#f87171', s=18, zorder=5,
           label=f'Anomalies (n={len(anom_b)})')
ax.set_ylabel('Indoor brightness (%)', fontsize=10)
ax.set_ylim(0, 105)
ax.set_title('Panel A — Indoor brightness: 2h moving average with anomaly detection', fontsize=10)
ax.legend(fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

ax2 = axes[1]
ax2.plot(hr_daily['date_dt'] if 'date_dt' in hr_daily.columns
         else pd.to_datetime(hr_daily['date_str']),
         hr_daily['hr_mean_night'],
         color='#f87171', linewidth=2, marker='o', markersize=6,
         label='Mean sleep HR (bpm)')
hr_daily['date_dt'] = pd.to_datetime(hr_daily['date_str'])
ax2.axhline(mean_hr, color='#94a3b8', linewidth=1.5, linestyle='--',
            label=f'Overall mean ({mean_hr:.1f} bpm)')
ax2.fill_between(hr_daily['date_dt'], lower_hr, upper_hr,
                 color='#f87171', alpha=0.12, label='±1.5 SD band')
for _, row in anomalies.iterrows():
    ax2.scatter(pd.to_datetime(row['date_str']), row['hr_mean_night'],
                color='#f59e0b', s=100, zorder=5, marker='*')
    ax2.annotate(f"score={int(row['sleep_score'])}",
                 (pd.to_datetime(row['date_str']), row['hr_mean_night']),
                 textcoords='offset points', xytext=(6, 6),
                 fontsize=9, color='#f59e0b')
ax2.set_ylabel('Mean sleep HR (bpm)', fontsize=10)
ax2.set_title('Panel B — Per-night mean sleep HR with anomaly detection (±1.5 SD)', fontsize=10)
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(axis='y', alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.tight_layout()
plt.savefig('figures/fig5_moving_average_anomaly.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved fig5_moving_average_anomaly")

print()
print("=" * 60)
print("ALL DONE")
print("=" * 60)