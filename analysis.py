import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

from data_prep import merged, ts_aligned

os.makedirs('figures', exist_ok=True)
merged['date_dt'] = pd.to_datetime(merged['light_date'])

SOLAR  = '#f59e0b'
BRIGHT = '#a78bfa'
DEEP   = '#818cf8'
REM    = '#60a5fa'
LIGHT  = '#cbd5e1'
HUM_DAY   = '#38bdf8'
HUM_PRE   = '#818cf8'
HUM_SLEEP = '#60a5fa'
HR     = '#f87171'
CYAN   = '#06b6d4'


def corr(x, y, label=''):
    sub = merged[[x, y]].dropna()
    r, p = stats.pearsonr(sub[x], sub[y])
    sig = '**' if p < 0.05 else ('*' if p < 0.1 else 'ns')
    print(f"  [{sig}]  r={r:+.3f}  p={p:.3f}  n={len(sub)}  {label}")
    return r, p


print("solar / cloud -> brightness")
corr('avg_solar', 'avg_brightness', 'solar -> brightness')
corr('avg_cloud', 'avg_brightness', 'cloud -> brightness')
sub = merged[['avg_solar', 'avg_brightness']].dropna()
slope, intercept, r, p, _ = stats.linregress(sub['avg_solar'], sub['avg_brightness'])
print(f"  OLS: brightness = {slope:.3f} x solar + {intercept:.2f},  R2={r**2:.3f}")

print("\nbrightness -> sleep (lag 1d)")
for tgt, lbl in [
    ('sleep_score', 'sleep score'),
    ('deep_pct', 'deep %'),
    ('rem_pct', 'REM %'),
    ('total_sleep_h', 'total sleep h'),
    ('hr_mean', 'mean HR'),
    ('resp_mean', 'mean resp'),
    ('resp_std', 'resp variability'),
    ('total_awake_min', 'awake min'),
    ('first_deep_min', 'time to first deep'),
    ('n_transitions', 'stage transitions'),
]:
    corr('avg_brightness', tgt, f'brightness -> {lbl}')

print("\nsolar -> sleep")
for tgt, lbl in [('sleep_score', 'sleep score'), ('deep_pct', 'deep %'), ('hr_mean', 'mean HR')]:
    corr('avg_solar', tgt, f'solar -> {lbl}')

print("\nhumidity -> sleep")
for hum, hlbl in [
    ('avg_humidity_day', 'daytime (08-18)'),
    ('avg_humidity_presleep', 'pre-sleep (20-23)'),
    ('avg_humidity_sleep', 'sleep period'),
]:
    print(f"  {hlbl}")
    for tgt, lbl in [('sleep_score', 'sleep score'), ('hr_mean', 'mean HR'), ('deep_pct', 'deep %')]:
        corr(hum, tgt, lbl)

print("\ntemp -> sleep")
for tgt, lbl in [('sleep_score', 'sleep score'), ('deep_pct', 'deep %'), ('hr_mean', 'mean HR')]:
    corr('avg_temp', tgt, lbl)

print("\nconfounding: outdoor humidity")
corr('outdoor_humidity', 'sleep_score', 'outdoor humidity -> sleep score')
corr('outdoor_humidity', 'avg_brightness', 'outdoor humidity -> brightness')


r_ts, _ = stats.pearsonr(ts_aligned['solar_radiation'], ts_aligned['brightness_pct'])

fig, ax = plt.subplots(figsize=(14, 5))
ax2 = ax.twinx()
ax.fill_between(ts_aligned['timestamp'], ts_aligned['solar_radiation'], color=SOLAR, alpha=0.3, label='Solar radiation (W/m²)')
ax.plot(ts_aligned['timestamp'], ts_aligned['solar_radiation'], color=SOLAR, linewidth=0.8, alpha=0.9)
ax2.fill_between(ts_aligned['timestamp'], ts_aligned['brightness_pct'], color=BRIGHT, alpha=0.3, label='Indoor brightness (%)')
ax2.plot(ts_aligned['timestamp'], ts_aligned['brightness_pct'], color=BRIGHT, linewidth=0.8, alpha=0.9)
ax.set_ylabel('Solar radiation (W/m²)', color=SOLAR, fontsize=10)
ax2.set_ylabel('Indoor brightness (%)', color=BRIGHT, fontsize=10)
ax.set_ylim(0, 550)
ax2.set_ylim(0, 105)
ax.set_title(f'Outdoor Solar Radiation vs Indoor Brightness — 15-min Time Series\n'
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
print("saved fig0_solar_vs_brightness_15min")


fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Indoor Daylighting & Sleep — Daily Overview\nLondon, Feb 2026',
             fontsize=13, fontweight='bold', y=0.98)

ax = axes[0]
axb = ax.twinx()
ax.bar(merged['date_dt'], merged['avg_solar'], color=SOLAR, alpha=0.55, width=0.7, label='Solar (W/m²)')
axb.plot(merged['date_dt'], merged['avg_brightness'], color=BRIGHT, marker='o', linewidth=2, markersize=5, label='Brightness (%)')
ax.set_ylabel('Solar radiation (W/m²)', color=SOLAR, fontsize=10)
axb.set_ylabel('Indoor brightness (%)', color=BRIGHT, fontsize=10)
ax.set_ylim(0, 400)
axb.set_ylim(0, 80)
lines = ax.get_legend_handles_labels()[0] + axb.get_legend_handles_labels()[0]
labels = ax.get_legend_handles_labels()[1] + axb.get_legend_handles_labels()[1]
ax.legend(lines, labels, loc='upper right', fontsize=9)
ax.set_title('Panel A — Daily mean solar vs indoor brightness (08:00–18:00)', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.text(0.01, 0.88, 'r = +0.838, p < 0.001', transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffbe6', alpha=0.8))

ax = axes[1]
score_colors = ['#34d399' if s >= 80 else '#fbbf24' if s >= 65 else '#f87171' for s in merged['sleep_score']]
ax.bar(merged['date_dt'], merged['sleep_score'], color=score_colors, alpha=0.75, width=0.7)
ax.axhline(80, color='#34d399', linewidth=1, linestyle='--', alpha=0.5)
ax.axhline(65, color='#fbbf24', linewidth=1, linestyle='--', alpha=0.5)
axb = ax.twinx()
axb.plot(merged['date_dt'], merged['avg_brightness'], color=BRIGHT, linewidth=1.5, linestyle=':', alpha=0.6)
ax.set_ylabel('Sleep score', fontsize=10)
axb.set_ylabel('Indoor brightness (%)', color=BRIGHT, fontsize=10)
ax.set_ylim(40, 100)
axb.set_ylim(0, 80)
ax.set_title('Panel B — Sleep score (green ≥80, yellow 65–79, red <65)', fontsize=10)
ax.grid(axis='y', alpha=0.3)

ax = axes[2]
ax.stackplot(merged['date_dt'],
             merged['deep_pct'], merged['rem_pct'],
             100 - merged['deep_pct'] - merged['rem_pct'] - merged['awake_h'] / merged['total_sleep_h'] * 100,
             labels=['Deep', 'REM', 'Light'],
             colors=[DEEP, REM, LIGHT], alpha=0.85)
ax.set_ylabel('Sleep composition (%)', fontsize=10)
ax.set_title('Panel C — Sleep stage composition', fontsize=10)
ax.set_ylim(0, 100)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.tight_layout()
plt.savefig('figures/fig1_daily_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved fig1_daily_overview")


fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle('Correlation Analysis — Daylighting & Sleep Quality', fontsize=13, fontweight='bold')

def scatter_fit(ax, x, y, xlabel, ylabel, title, color):
    sub = merged[[x, y]].dropna()
    ax.scatter(sub[x], sub[y], color=color, s=55, alpha=0.85, edgecolors='white', linewidth=0.5, zorder=3)
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

scatter_fit(axes[0, 0], 'avg_solar', 'avg_brightness', 'Solar radiation (W/m²)', 'Indoor brightness (%)', 'A  Solar -> Brightness', SOLAR)
scatter_fit(axes[0, 1], 'avg_brightness', 'deep_pct', 'Indoor brightness (%)', 'Deep sleep (%)', 'B  Brightness -> Deep sleep', DEEP)
scatter_fit(axes[0, 2], 'avg_brightness', 'resp_std', 'Indoor brightness (%)', 'Respiration variability (SD, br/min)', 'C  Brightness -> Resp variability', CYAN)
scatter_fit(axes[1, 0], 'avg_humidity_presleep', 'sleep_score', 'Pre-sleep humidity (%, 20-23h)', 'Sleep score', 'D  Humidity -> Sleep score', HUM_PRE)
scatter_fit(axes[1, 1], 'avg_humidity_presleep', 'hr_mean', 'Pre-sleep humidity (%, 20-23h)', 'Mean sleep HR (bpm)', 'E  Humidity -> Sleep HR', HR)
scatter_fit(axes[1, 2], 'avg_solar', 'deep_pct', 'Solar radiation (W/m²)', 'Deep sleep (%)', 'F  Solar -> Deep sleep', SOLAR)

plt.tight_layout()
plt.savefig('figures/fig2_scatter_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved fig2_scatter_correlations")


fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle('Humidity Window Comparison', fontsize=12, fontweight='bold')

for ax, (hum, hlbl, color) in zip(axes, [
    ('avg_humidity_day',      'Daytime (08:00-18:00)',   HUM_DAY),
    ('avg_humidity_presleep', 'Pre-sleep (20:00-23:00)', HUM_PRE),
    ('avg_humidity_sleep',    'Sleep period',            HUM_SLEEP),
]):
    sub = merged[[hum, 'sleep_score']].dropna()
    ax.scatter(sub[hum], sub['sleep_score'], color=color, s=55, alpha=0.85, edgecolors='white', linewidth=0.5)
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
plt.savefig('figures/fig3_humidity_windows.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved fig3_humidity_windows")


predictors = ['avg_brightness', 'avg_solar', 'avg_humidity_presleep', 'avg_temp']
outcomes = ['sleep_score', 'deep_pct', 'rem_pct', 'hr_mean', 'resp_mean', 'resp_std']
pred_labels = ['Indoor\nbrightness', 'Outdoor\nsolar', 'Pre-sleep\nhumidity', 'Indoor\ntemp']
out_labels = ['Sleep\nscore', 'Deep\nsleep %', 'REM %', 'Sleep\nHR', 'Resp\nmean', 'Resp\nvariability']

matrix = np.zeros((len(predictors), len(outcomes)))
pmatrix = np.ones((len(predictors), len(outcomes)))
for i, pred in enumerate(predictors):
    for j, out in enumerate(outcomes):
        sub = merged[[pred, out]].dropna()
        if len(sub) >= 5:
            r, p = stats.pearsonr(sub[pred], sub[out])
            matrix[i, j] = r
            pmatrix[i, j] = p

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle('Pearson r — Environmental predictors vs sleep outcomes (lag=1d)', fontsize=11, fontweight='bold')
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
fig.text(0.02, 0.01,
         'REM %: REM sleep as % of total sleep  |  Sleep HR: mean heart rate during sleep (bpm)  |  '
         'Resp mean: mean respiration rate (br/min)  |  Resp variability: SD of respiration rate',
         fontsize=7.5, color='#666', style='italic')
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('figures/fig4_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved fig4_correlation_heatmap")