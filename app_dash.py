import sys, os
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_dp", os.path.join(_dir, "data_prep.py"))
_dp = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_dp)
merged = _dp.merged
ts_aligned = _dp.ts_aligned

import numpy as np
import pandas as pd
from scipy import stats
from dash import Dash, html, dcc, Input, Output, State, callback
import dash
import plotly.graph_objects as go
from plotly.subplots import make_subplots

merged = merged[merged['light_date'] <= '2026-02-28'].copy()
ts_aligned = ts_aligned[pd.to_datetime(ts_aligned['timestamp']).dt.date <= pd.Timestamp('2026-02-28').date()].copy()

indoor_raw = pd.read_excel('indoor_data.xlsx', sheet_name='indoor')
indoor_raw = indoor_raw.drop(columns=['Unnamed: 4'], errors='ignore')
indoor_raw['timestamp'] = pd.to_datetime(indoor_raw['timestamp'])
hr_raw   = pd.read_csv('sleep_hr_timeseries.csv')
resp_raw = pd.read_csv('sleep_respiration_timeseries.csv')
hr_raw['ts']   = pd.to_datetime(hr_raw['timestamp'], unit='ms')
resp_raw['ts'] = pd.to_datetime(resp_raw['timestamp'], unit='ms')
sleep_raw = pd.read_excel('sleep_summary.xlsx', sheet_name='Sheet1')
sleep_raw['sleep_start_dt'] = pd.to_datetime(sleep_raw['sleep_start'], unit='ms')
sleep_raw['sleep_end_dt']   = pd.to_datetime(sleep_raw['sleep_end'],   unit='ms')
sleep_raw['date_str'] = pd.to_datetime(sleep_raw['date']).dt.date.astype(str)
valid_nights = sleep_raw[
    (sleep_raw['sleep_start_dt'] >= pd.Timestamp('2026-02-17')) &
    (sleep_raw['sleep_end_dt']   <  pd.Timestamp('2026-03-01'))
].copy()

BG     = '#0d0d0f'
BG2    = '#141416'
BG3    = '#1a1a1d'
BORDER = '#242428'
TEXT   = '#e8e6f0'
MUTED  = '#70707a'
SOLAR  = '#f59e0b'
BRIGHT = '#a78bfa'
DEEP   = '#818cf8'
REM    = '#60a5fa'
GREEN  = '#4ade80'
YELLOW = '#fbbf24'
RED    = '#f87171'
HUM    = '#38bdf8'

GRID   = dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.08)',
              zerolinecolor='rgba(0,0,0,0)', showgrid=True, zeroline=False)
NOGRID = dict(showgrid=False, linecolor='rgba(255,255,255,0.08)', zeroline=False)

def plot_layout(**kwargs):
    base = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=MUTED, size=11))
    base.update(kwargs)
    return base

def card(title, children, extra_style=None):
    style = {'background': BG2, 'border': f'1px solid {BORDER}', 'borderRadius': '12px',
             'padding': '18px 20px', 'marginBottom': '14px'}
    if extra_style:
        style.update(extra_style)
    return html.Div([
        html.Div(title, style={'fontSize': '10px', 'letterSpacing': '.8px', 'textTransform': 'uppercase',
                               'color': MUTED, 'fontFamily': 'monospace', 'marginBottom': '14px',
                               'paddingBottom': '8px', 'borderBottom': f'1px solid {BORDER}'}),
        *children
    ], style=style)

def stat_row(label, value, value_color=TEXT):
    return html.Div([
        html.Span(label, style={'color': MUTED, 'fontSize': '12px'}),
        html.Span(value, style={'fontWeight': '600', 'fontFamily': 'monospace',
                                'fontSize': '12px', 'color': value_color}),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
              'padding': '5px 0', 'borderBottom': f'1px solid {BG3}'})

def badge(text, score):
    if score >= 80:   bg, col = 'rgba(74,222,128,.15)',  GREEN
    elif score >= 65: bg, col = 'rgba(251,191,36,.15)',  YELLOW
    else:             bg, col = 'rgba(248,113,113,.15)', RED
    return html.Span(str(text), style={'background': bg, 'color': col, 'padding': '2px 8px',
                                       'borderRadius': '4px', 'fontSize': '11px',
                                       'fontFamily': 'monospace', 'fontWeight': '500'})


# precompute
r_sb, _ = stats.pearsonr(merged['avg_solar'], merged['avg_brightness'])
r_ts, _ = stats.pearsonr(ts_aligned['solar_radiation'], ts_aligned['brightness_pct'])
best  = merged.loc[merged['sleep_score'].idxmax()]
worst = merged.loc[merged['sleep_score'].idxmin()]
peak_date = merged.loc[merged['avg_solar'].idxmax(), 'light_date'][5:]
sub_hs = merged[['avg_humidity_sleep','sleep_score']].dropna()
high_hum = sub_hs[sub_hs['avg_humidity_sleep'] > 35]
low_hum  = sub_hs[sub_hs['avg_humidity_sleep'] <= 35]
bright_days = merged[merged['avg_brightness'] > 25]
dim_days    = merged[merged['avg_brightness'] <= 25]

# key insights precompute
_brightness_median = merged['avg_brightness'].median()
_bright = merged[merged['avg_brightness'] >  _brightness_median]
_dim    = merged[merged['avg_brightness'] <= _brightness_median]
_ki_bright_deep = _bright['deep_pct'].mean()
_ki_dim_deep    = _dim['deep_pct'].mean()
_sub_hs = merged[['avg_humidity_sleep','sleep_score']].dropna()
_ki_hum_high = _sub_hs[_sub_hs['avg_humidity_sleep'] >  35]['sleep_score'].mean()
_ki_hum_low  = _sub_hs[_sub_hs['avg_humidity_sleep'] <= 35]['sleep_score'].mean()
_r_bright_deep, _p_bright_deep = stats.pearsonr(
    merged[['avg_brightness','deep_pct']].dropna()['avg_brightness'],
    merged[['avg_brightness','deep_pct']].dropna()['deep_pct'])
_r_hum_score, _p_hum_score = stats.pearsonr(_sub_hs['avg_humidity_sleep'], _sub_hs['sleep_score'])

CORRS = [
    ('Solar → Indoor brightness',    stats.pearsonr(merged['avg_solar'], merged['avg_brightness'])),
    ('Brightness → Deep sleep %',    stats.pearsonr(merged['avg_brightness'], merged['deep_pct'])),
    ('Brightness → Resp variability',stats.pearsonr(merged['avg_brightness'], merged['resp_std'])),
    ('Sleep humidity → Sleep score', stats.pearsonr(*[sub_hs[c].values for c in ['avg_humidity_sleep','sleep_score']])),
    ('Sleep humidity → Sleep HR',    stats.pearsonr(*[merged[['avg_humidity_sleep','hr_mean']].dropna()[c].values for c in ['avg_humidity_sleep','hr_mean']])),
    ('Sleep humidity → Resp rate',   stats.pearsonr(*[merged[['avg_humidity_sleep','resp_mean']].dropna()[c].values for c in ['avg_humidity_sleep','resp_mean']])),
]

def corr_bar_row(label, r, p):
    sig = '**' if p < 0.05 else ('*' if p < 0.1 else 'ns')
    bar_col = '#f87171' if r < 0 else '#60a5fa'
    return html.Div([
        html.Div(label, style={'width': '240px', 'fontSize': '12px', 'flexShrink': '0',
                               'color': TEXT, 'fontWeight': '400'}),
        html.Div(html.Div(style={'height': '100%', 'width': f'{abs(r)*100:.0f}%',
                         'backgroundColor': bar_col, 'borderRadius': '3px'}),
                 style={'flex': '1', 'background': BG3, 'height': '6px',
                        'borderRadius': '3px', 'overflow': 'hidden', 'alignSelf': 'center'}),
        html.Div(f'{r:+.3f}', style={'width': '52px', 'textAlign': 'right',
                                     'fontFamily': 'monospace', 'fontSize': '12px', 'color': bar_col}),
        html.Div([
            html.Span(sig, style={'color': TEXT if sig != 'ns' else MUTED, 'fontSize': '11px', 'display': 'block'}),
            html.Span(f'p={p:.3f}', style={'color': MUTED, 'fontSize': '10px', 'fontFamily': 'monospace'}),
        ], style={'width': '65px', 'textAlign': 'right'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '12px', 'marginBottom': '10px'})

# ts chart
solar_sm  = ts_aligned['solar_radiation'].rolling(3, center=True, min_periods=1).mean()
bright_sm = ts_aligned['brightness_pct'].rolling(3, center=True, min_periods=1).mean()

fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
fig_ts.add_trace(go.Scatter(
    x=ts_aligned['timestamp'], y=solar_sm,
    name='Solar radiation (W/m²)',
    fill='tozeroy',
    line=dict(color=SOLAR, width=1.5, shape='spline', smoothing=1.3),
    fillcolor='rgba(245,158,11,0.15)',
    customdata=ts_aligned['solar_radiation'],
    hovertemplate='%{x|%b %d %H:%M}<br>Solar: %{customdata:.0f} W/m²<extra></extra>'
), secondary_y=False)
fig_ts.add_trace(go.Scatter(
    x=ts_aligned['timestamp'], y=bright_sm,
    name='Indoor brightness (%)',
    fill='tozeroy',
    line=dict(color=BRIGHT, width=1.5, shape='spline', smoothing=1.3),
    fillcolor='rgba(167,139,250,0.15)',
    customdata=ts_aligned['brightness_pct'],
    hovertemplate='%{x|%b %d %H:%M}<br>Brightness: %{customdata:.1f}%<extra></extra>'
), secondary_y=True)
fig_ts.update_layout(**plot_layout(
    height=240, hovermode='x unified',
    legend=dict(orientation='h', y=1.15, font=dict(color=TEXT, size=11)),
    margin=dict(t=10, b=40, l=60, r=70),
    hoverlabel=dict(bgcolor=BG2, bordercolor=BORDER, font=dict(color=TEXT, size=11))
))
fig_ts.update_yaxes(title_text='W/m²', secondary_y=False,
    title_font=dict(color=SOLAR, size=10), tickfont=dict(color=MUTED), **GRID)
fig_ts.update_yaxes(title_text='Brightness (%)', secondary_y=True,
    title_font=dict(color=BRIGHT, size=10), tickfont=dict(color=MUTED),
    range=[0, 100], showgrid=False)
fig_ts.update_xaxes(tickformat='%b %d', dtick=86400000*2, **NOGRID, tickfont=dict(color=TEXT, size=11))

def make_scatter_fig(x, y, xlabel, ylabel, color, title, date_col='light_date'):
    sub = merged[[x, y]].dropna()
    dates = merged.loc[sub.index, date_col].str[5:]
    r, p = stats.pearsonr(sub[x], sub[y])
    m, b, _, _, _ = stats.linregress(sub[x], sub[y])
    xline = np.linspace(sub[x].min(), sub[x].max(), 100)
    sig = '**' if p < 0.05 else ('*' if p < 0.1 else '')
    from scipy.ndimage import gaussian_filter1d
    ysmooth = gaussian_filter1d(m*xline+b, sigma=1)
    fig = go.Figure()
    try:
        fc = f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)'
    except Exception:
        fc = 'rgba(100,100,200,0.06)'
    fig.add_trace(go.Scatter(x=xline, y=ysmooth, mode='lines',
        line=dict(color=color, width=0), fill='tozeroy',
        fillcolor=fc, showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=xline, y=ysmooth, mode='lines',
        line=dict(color=color, width=2, dash='dot'), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=sub[x], y=sub[y], mode='markers',
        marker=dict(color=color, size=10, line=dict(color='white', width=1.5), opacity=0.9),
        customdata=dates,
        hovertemplate=f'<b>%{{customdata}}</b><br>{xlabel}: %{{x:.1f}}<br>{ylabel}: %{{y:.1f}}<extra></extra>'))
    y_min = sub[y].min()
    y_max = sub[y].max()
    y_pad = (y_max - y_min) * 0.15
    fig.update_layout(**plot_layout(
        height=300,
        title=dict(text=f'{title}<br><sup>r={r:+.3f}{sig}  p={p:.3f}  n={len(sub)}</sup>',
                   font=dict(size=12, color=TEXT)),
        xaxis=dict(title=xlabel, **NOGRID, rangemode='normal'),
        yaxis=dict(title=ylabel, **GRID,
                   range=[max(-2, y_min - y_pad * 2), y_max + y_pad]),
        showlegend=False,
        margin=dict(t=60, b=40, l=55, r=20),
        hoverlabel=dict(bgcolor=BG2, bordercolor=BORDER, font=dict(color=TEXT, size=11))))
    return fig, r, p

fig_s1, r1, p1 = make_scatter_fig('avg_solar', 'avg_brightness', 'Solar radiation (W/m²)', 'Indoor brightness (%)', DEEP, 'Solar radiation vs Indoor brightness')
fig_s2, r2, p2 = make_scatter_fig('avg_brightness', 'deep_pct', 'Indoor brightness (%)', 'Deep sleep (%)', REM, 'Indoor brightness vs Deep sleep % (lag=1d)')
fig_s3, r3, p3 = make_scatter_fig('avg_brightness', 'resp_std', 'Indoor brightness (%)', 'Resp variability (SD)', '#06b6d4', 'Indoor brightness vs Resp variability (lag=1d)')

def make_hum_fig(y_var, y_label, color=DEEP):
    sub = merged[['avg_humidity_sleep', y_var, 'date_str']].dropna()
    r, p = stats.pearsonr(sub['avg_humidity_sleep'], sub[y_var])
    m, b, _, _, _ = stats.linregress(sub['avg_humidity_sleep'], sub[y_var])
    xline = np.linspace(sub['avg_humidity_sleep'].min(), sub['avg_humidity_sleep'].max(), 100)
    sig = '**' if p < 0.05 else ('*' if p < 0.1 else '')
    from scipy.ndimage import gaussian_filter1d
    ysmooth = gaussian_filter1d(m*xline+b, sigma=1)
    try:
        fc = f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)'
    except Exception:
        fc = 'rgba(100,100,200,0.06)'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xline, y=ysmooth, mode='lines',
        line=dict(color=color, width=0), fill='tozeroy',
        fillcolor=fc, showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=xline, y=ysmooth, mode='lines',
        line=dict(color=color, width=2, dash='dot'), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=sub['avg_humidity_sleep'], y=sub[y_var], mode='markers',
        marker=dict(color=color, size=10, line=dict(color='white', width=1.5), opacity=0.9),
        customdata=sub['date_str'].str[5:],
        hovertemplate=f'<b>%{{customdata}}</b><br>Humidity: %{{x:.1f}}%<br>{y_label}: %{{y:.1f}}<extra></extra>'))
    hy_min = sub[y_var].min()
    hy_max = sub[y_var].max()
    hy_pad = (hy_max - hy_min) * 0.2
    fig.update_layout(**plot_layout(
        height=280,
        title=dict(text=f'Sleep-period humidity vs {y_label}<br><sup>r={r:+.3f}{sig}  p={p:.3f}  n={len(sub)}</sup>',
                   font=dict(size=12, color=TEXT)),
        xaxis=dict(title='Sleep-period humidity (%)', **NOGRID, rangemode='normal'),
        yaxis=dict(title=y_label, **GRID,
                   range=[max(0, hy_min - hy_pad), hy_max + hy_pad]),
        showlegend=False,
        margin=dict(t=60, b=40, l=55, r=20),
        hoverlabel=dict(bgcolor=BG2, bordercolor=BORDER, font=dict(color=TEXT, size=11))))
    return fig, r, p

fig_h1, rh1, ph1 = make_hum_fig('sleep_score', 'Sleep score', DEEP)
fig_h2, rh2, ph2 = make_hum_fig('hr_mean', 'Mean sleep HR (bpm)', REM)
fig_h3, rh3, ph3 = make_hum_fig('resp_mean', 'Resp rate (br/min)', '#06b6d4')

# sleep overview charts
score_idx = list(range(len(merged)))
score_labels = merged['date_str'].str[5:].tolist()
score_colors = ['rgba(129,140,248,0.85)' for _ in merged['sleep_score']]
fig_score = make_subplots(specs=[[{"secondary_y": True}]])
fig_score.add_trace(go.Bar(
    x=score_idx, y=merged['sleep_score'].tolist(),
    name='Sleep score', marker_color=score_colors, marker_line_width=0,
    text=merged['sleep_score'].tolist(),
    textposition='outside', textfont=dict(size=10, color=TEXT)), secondary_y=False)
fig_score.add_trace(go.Scatter(
    x=score_idx, y=merged['avg_brightness'].tolist(),
    name='Brightness (%)', mode='lines+markers',
    line=dict(color=BRIGHT, width=1.5, dash='dot'),
    marker=dict(size=4, color=BRIGHT)), secondary_y=True)
fig_score.add_hline(y=80, line_dash='dot', line_color='rgba(255,255,255,0.15)', secondary_y=False)
fig_score.add_hline(y=65, line_dash='dot', line_color='rgba(255,255,255,0.08)', secondary_y=False)
fig_score.update_layout(**plot_layout(
    height=220,
    legend=dict(orientation='h', y=1.15, font=dict(color=TEXT, size=10)),
    margin=dict(t=10, b=30, l=50, r=60),
    bargap=0.35))
fig_score.update_yaxes(range=[40,100], secondary_y=False, title_text='Sleep score',
    title_font=dict(size=10, color=MUTED), tickfont=dict(color=MUTED), **GRID)
fig_score.update_yaxes(range=[0,80], secondary_y=True, title_text='Brightness (%)',
    title_font=dict(color=BRIGHT, size=10), tickfont=dict(color=BRIGHT), showgrid=False)
fig_score.update_xaxes(**NOGRID, tickfont=dict(color=TEXT, size=10),
    tickmode='array', tickvals=score_idx[::2], ticktext=score_labels[::2])

light_only = 100 - merged['deep_pct'] - merged['rem_pct'] - merged['awake_h']/merged['total_sleep_h']*100
stage_x = list(range(len(merged)))
stage_labels = merged['date_str'].str[5:].tolist()
fig_stages = go.Figure()
fig_stages.add_trace(go.Scatter(
    x=stage_x, y=merged['deep_pct'].tolist(),
    name='Deep', fill='tozeroy', mode='lines',
    line=dict(color='#818cf8', width=1, shape='spline', smoothing=0.8),
    fillcolor='rgba(129,140,248,0.85)',
    hovertemplate='%{text}<br>Deep: %{y:.1f}%<extra></extra>',
    text=stage_labels))
fig_stages.add_trace(go.Scatter(
    x=stage_x, y=(merged['deep_pct']+merged['rem_pct']).tolist(),
    name='REM', fill='tonexty', mode='lines',
    line=dict(color='#60a5fa', width=1, shape='spline', smoothing=0.8),
    fillcolor='rgba(96,165,250,0.65)',
    hovertemplate='%{text}<br>REM: %{y:.1f}%<extra></extra>',
    text=stage_labels))
fig_stages.add_trace(go.Scatter(
    x=stage_x, y=(merged['deep_pct']+merged['rem_pct']+light_only).tolist(),
    name='Light', fill='tonexty', mode='lines',
    line=dict(color='#94a3b8', width=1, shape='spline', smoothing=0.8),
    fillcolor='rgba(148,163,184,0.35)',
    hovertemplate='%{text}<br>Light: %{y:.1f}%<extra></extra>',
    text=stage_labels))
fig_stages.update_layout(**plot_layout(
    height=220,
    legend=dict(orientation='h', y=1.15, font=dict(color=TEXT, size=10)),
    yaxis=dict(range=[0,100], title='%', **GRID, tickfont=dict(color=MUTED),
               title_font=dict(color=MUTED, size=10)),
    xaxis=dict(**NOGRID, tickfont=dict(color=TEXT, size=10),
               tickmode='array', tickvals=stage_x[::2],
               ticktext=stage_labels[::2]),
    margin=dict(t=10, b=30, l=40, r=10),
    hoverlabel=dict(bgcolor=BG2, bordercolor=BORDER, font=dict(color=TEXT, size=11))))

NAV_STYLE = {'padding': '8px 14px', 'borderRadius': '6px', 'cursor': 'pointer',
             'fontSize': '12px', 'color': MUTED, 'fontFamily': 'monospace',
             'letterSpacing': '.5px', 'border': 'none', 'background': 'none',
             'textDecoration': 'none', 'display': 'block', 'marginBottom': '2px'}

app = Dash(__name__, suppress_callback_exceptions=True)
app.index_string = app.index_string.replace('</head>', """
<style>
/* Dash React-Select v2 actual class names */
.dark-dropdown .Select-control,
.dark-dropdown .Select--single > .Select-control {
    background-color: #1a1a1d !important;
    border: 1px solid #242428 !important;
    border-radius: 6px !important;
    color: #e8e6f0 !important;
}
.dark-dropdown .Select-control:hover { border-color: #a78bfa !important; }
.dark-dropdown .Select.is-focused > .Select-control {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 1px rgba(167,139,250,0.25) !important;
}
.dark-dropdown .Select-menu-outer {
    background-color: #1a1a1d !important;
    border: 1px solid #242428 !important;
    border-radius: 0 0 6px 6px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.8) !important;
    z-index: 9999 !important;
}
.dark-dropdown .Select-option {
    background-color: #1a1a1d !important;
    color: #70707a !important;
    font-family: monospace !important;
    font-size: 12px !important;
}
.dark-dropdown .Select-option.is-focused,
.dark-dropdown .Select-option:hover {
    background-color: #242428 !important;
    color: #e8e6f0 !important;
}
.dark-dropdown .Select-option.is-selected {
    background-color: #242428 !important;
    color: #a78bfa !important;
}
.dark-dropdown .Select-value-label,
.dark-dropdown .Select-placeholder { color: #e8e6f0 !important; font-family: monospace !important; font-size: 12px !important; }
.dark-dropdown .Select-arrow { border-top-color: #70707a !important; }
.dark-dropdown .Select-input > input { color: #e8e6f0 !important; }
.dark-dropdown .Select-noresults { background: #1a1a1d !important; color: #70707a !important; }
/* ── custom range slider ── */
input[type=range] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 4px;
    border-radius: 2px;
    outline: none;
    cursor: pointer;
    background: transparent;
}
/* track — webkit */
input[type=range]::-webkit-slider-runnable-track {
    height: 4px;
    border-radius: 2px;
    background: #242428;
}
/* track — firefox */
input[type=range]::-moz-range-track {
    height: 4px;
    border-radius: 2px;
    background: #242428;
}
/* fill left of thumb — firefox */
input[type=range]::-moz-range-progress {
    height: 4px;
    border-radius: 2px;
    background: #a78bfa;
}
/* thumb — webkit */
input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #a78bfa;
    border: 2px solid #e8e6f0;
    margin-top: -6px;
    box-shadow: 0 0 0 0 rgba(167,139,250,0.3);
    transition: box-shadow 0.15s ease;
}
input[type=range]:hover::-webkit-slider-thumb,
input[type=range]:focus::-webkit-slider-thumb {
    box-shadow: 0 0 0 4px rgba(167,139,250,0.25);
}
/* thumb — firefox */
input[type=range]::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #a78bfa;
    border: 2px solid #e8e6f0;
    cursor: pointer;
}

/* slider tooltip */
.rc-slider-tooltip-inner {
    background-color: #1a1a1d !important;
    border: 1px solid #242428 !important;
    color: #e8e6f0 !important;
    font-family: monospace !important;
    font-size: 11px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.6) !important;
    padding: 3px 10px !important;
    border-radius: 4px !important;
    min-width: 28px !important;
}
.rc-slider-tooltip-arrow { border-top-color: #242428 !important; }
</style>
""")


app.layout = html.Div(style={'background': BG, 'minHeight': '100vh',
                              'fontFamily': 'DM Sans, system-ui, sans-serif', 'color': TEXT}, children=[
    html.Div(style={'display': 'flex', 'maxWidth': '1200px', 'margin': '0 auto'}, children=[
        # sidebar
        html.Div(style={'width': '200px', 'flexShrink': '0', 'background': '#0f1014',
                        'borderRight': f'1px solid {BORDER}', 'padding': '24px 16px',
                        'minHeight': '100vh', 'position': 'sticky', 'top': '0'}, children=[
            html.Div('Indoor Environment', style={'fontSize': '13px', 'fontWeight': '500', 'color': TEXT, 'lineHeight': '1.4'}),
            html.Div('& Sleep Quality', style={'fontSize': '13px', 'fontWeight': '500', 'color': TEXT, 'marginBottom': '4px'}),
            html.Div('ELEC70126 · Imperial College', style={'fontSize': '10px', 'color': MUTED, 'fontFamily': 'monospace',
                     'paddingBottom': '14px', 'borderBottom': f'1px solid {BORDER}', 'marginBottom': '12px'}),
            dcc.Link('Overview',         href='/',            style=NAV_STYLE),
            dcc.Link('Daylighting',      href='/daylighting', style=NAV_STYLE),
            dcc.Link('Humidity',         href='/humidity',    style=NAV_STYLE),
            html.Div(style={'borderTop': f'1px solid {BORDER}', 'marginTop': '16px', 'paddingTop': '14px',
                            'fontSize': '11px', 'color': MUTED, 'lineHeight': '2'}, children=[
                'London, UK · 16–28 Feb 2026', html.Br(),
                'ESP32 + LDR + DHT11', html.Br(),
                'Open-Meteo API · Garmin Venu',
            ]),
        ]),
        # main
        html.Div(style={'flex': '1', 'padding': '28px', 'minWidth': '0'}, children=[
            dcc.Location(id='url', refresh=False),

            html.Div(id='page-content'),
            dcc.RadioItems(
                id='sleep-night-selector',
                options=[{'label': d[5:], 'value': d} for d in merged['date_str'].tolist()],
                value=merged['date_str'].tolist()[0],
                style={'display': 'none'}
            ),
            html.P('Data: ESP32 LDR/DHT11 · Open-Meteo API · Garmin Venu · Pearson r, OLS · ELEC70126 · Imperial College London',
                   style={'fontSize': '10px', 'color': MUTED, 'textAlign': 'center', 'marginTop': '32px',
                          'borderTop': f'1px solid {BORDER}', 'paddingTop': '16px'}),
        ]),
    ])
])

def page_overview():
    return html.Div([
        card('key insights', [
            html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(3,1fr)', 'gap': '12px'}, children=[
                # card 1
                html.Div([
                    html.Div('SOLAR VS INDOOR BRIGHTNESS', style={'fontSize': '10px', 'color': MUTED,
                             'letterSpacing': '.8px', 'fontFamily': 'monospace', 'marginBottom': '10px'}),
                    html.Div(f'r = {r_sb:.3f}', style={'fontSize': '36px', 'fontWeight': '600', 'color': SOLAR, 'marginBottom': '8px'}),
                    html.P(f'Room brightness closely tracks outdoor solar (R² = {r_sb**2*100:.0f}%). '
                           f'London Feb cloud cover {merged["avg_cloud"].mean():.0f}% limits indoor mean to {merged["avg_brightness"].mean():.1f}%.',
                           style={'fontSize': '12px', 'color': MUTED, 'lineHeight': '1.6', 'margin': '0'}),
                ], style={'background': BG3, 'border': f'1px solid {BORDER}', 'borderRadius': '10px', 'padding': '16px'}),
                # card 2
                html.Div([
                    html.Div('BRIGHT VS DIM DAYS — DEEP SLEEP', style={'fontSize': '10px', 'color': MUTED,
                             'letterSpacing': '.8px', 'fontFamily': 'monospace', 'marginBottom': '10px'}),
                    html.Div([
                        html.Span(f'{_ki_bright_deep:.1f}%', style={'fontSize': '26px', 'fontWeight': '600', 'color': BRIGHT}),
                        html.Span(' vs ', style={'fontSize': '16px', 'color': MUTED, 'margin': '0 6px'}),
                        html.Span(f'{_ki_dim_deep:.1f}%', style={'fontSize': '26px', 'fontWeight': '600', 'color': BRIGHT}),
                    ], style={'marginBottom': '8px', 'lineHeight': '1'}),
                    html.P(f'On brighter days (above median {_brightness_median:.1f}%), deep sleep % was lower '
                           f'the following night (r = {_r_bright_deep:.3f}, p = {_p_bright_deep:.3f}).',
                           style={'fontSize': '12px', 'color': MUTED, 'lineHeight': '1.6', 'margin': '0'}),
                ], style={'background': BG3, 'border': f'1px solid {BORDER}', 'borderRadius': '10px', 'padding': '16px'}),
                # card 3
                html.Div([
                    html.Div('LOW VS HIGH SLEEP HUMIDITY', style={'fontSize': '10px', 'color': MUTED,
                             'letterSpacing': '.8px', 'fontFamily': 'monospace', 'marginBottom': '10px'}),
                    html.Div([
                        html.Span(f'{_ki_hum_low:.0f}', style={'fontSize': '26px', 'fontWeight': '600', 'color': HUM}),
                        html.Span(' vs ', style={'fontSize': '16px', 'color': MUTED, 'margin': '0 6px'}),
                        html.Span(f'{_ki_hum_high:.0f}', style={'fontSize': '26px', 'fontWeight': '600', 'color': HUM}),
                    ], style={'marginBottom': '8px', 'lineHeight': '1'}),
                    html.P(f'Avg sleep score: humidity up to 35% vs above 35%. '
                           f'Higher sleep-period humidity associated with lower sleep quality '
                           f'(r = {_r_hum_score:.3f}, p = {_p_hum_score:.3f}).',
                           style={'fontSize': '12px', 'color': MUTED, 'lineHeight': '1.6', 'margin': '0'}),
                ], style={'background': BG3, 'border': f'1px solid {BORDER}', 'borderRadius': '10px', 'padding': '16px'}),
            ]),
        ]),
        card('15-min time series — outdoor solar vs indoor brightness', [
            dcc.Graph(figure=fig_ts, config={'displayModeBar': False}),
            html.P(f'r = {r_ts:.3f}, p < 0.001, n = {len(ts_aligned)} points · daytime 08:00–18:00',
                   style={'fontSize': '10px', 'color': MUTED, 'fontStyle': 'italic', 'marginTop': '-10px'}),
        ]),
        card('correlation summary', [
            *[corr_bar_row(lbl, r, p) for lbl, (r, p) in CORRS],
            html.P('** p<0.05   * p<0.1   n=13 days (daylighting, lag=1d)   n=12 nights (humidity)',
                   style={'fontSize': '10px', 'color': MUTED, 'fontStyle': 'italic', 'marginTop': '8px'}),
        ]),
        card('best vs worst sleep — environment comparison', [
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '14px'}, children=[
                html.Div([
                    html.Div(f'Best night — sleep {best["date_str"][5:]} (daytime {best["light_date"][5:]})',
                             style={'fontSize': '10px', 'color': MUTED, 'textTransform': 'uppercase', 'letterSpacing': '.6px', 'fontFamily': 'monospace', 'marginBottom': '10px'}),
                    stat_row('Sleep score', badge(int(best['sleep_score']), int(best['sleep_score']))),
                    stat_row('Indoor brightness', f'{best["avg_brightness"]:.1f}%'),
                    stat_row('Solar radiation', f'{best["avg_solar"]:.0f} W/m²'),
                    stat_row('Sleep humidity', f'{best["avg_humidity_sleep"]:.1f}%' if not pd.isna(best.get('avg_humidity_sleep', float('nan'))) else 'N/A'),
                    stat_row('Deep sleep %', f'{best["deep_pct"]:.1f}%'),
                    stat_row('Mean sleep HR', f'{best["hr_mean"]:.0f} bpm'),
                    stat_row('Resp rate', f'{best["resp_mean"]:.1f} br/min'),
                ], style={'background': BG3, 'border': f'1px solid rgba(74,222,128,.35)', 'borderRadius': '10px', 'padding': '14px'}),
                html.Div([
                    html.Div(f'Worst night — sleep {worst["date_str"][5:]} (daytime {worst["light_date"][5:]})',
                             style={'fontSize': '10px', 'color': MUTED, 'textTransform': 'uppercase', 'letterSpacing': '.6px', 'fontFamily': 'monospace', 'marginBottom': '10px'}),
                    stat_row('Sleep score', badge(int(worst['sleep_score']), int(worst['sleep_score']))),
                    stat_row('Indoor brightness', f'{worst["avg_brightness"]:.1f}%'),
                    stat_row('Solar radiation', f'{worst["avg_solar"]:.0f} W/m²'),
                    stat_row('Sleep humidity', f'{worst["avg_humidity_sleep"]:.1f}%' if not pd.isna(worst.get('avg_humidity_sleep', float('nan'))) else 'N/A'),
                    stat_row('Deep sleep %', f'{worst["deep_pct"]:.1f}%'),
                    stat_row('Mean sleep HR', f'{worst["hr_mean"]:.0f} bpm'),
                    stat_row('Resp rate', f'{worst["resp_mean"]:.1f} br/min'),
                ], style={'background': BG3, 'border': f'1px solid rgba(248,113,113,.35)', 'borderRadius': '10px', 'padding': '14px'}),
            ]),
        ]),
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '12px', 'marginBottom': '14px'}, children=[
            card('sleep score by night · brightness overlay', [
                dcc.Graph(figure=fig_score, config={'displayModeBar': False}, style={'height': '220px'})
            ], extra_style={'marginBottom': '0'}),
            card('sleep stage composition (%)', [
                dcc.Graph(figure=fig_stages, config={'displayModeBar': False}, style={'height': '220px'})
            ], extra_style={'marginBottom': '0'}),
        ]),
        card('select night', [
            html.Div([
                html.Button(
                    d[5:], id=f'nbtn-{d}', n_clicks=0,
                    style={'padding': '5px 10px', 'borderRadius': '6px',
                           'border': f'1px solid {BORDER}', 'background': BG3,
                           'color': MUTED, 'cursor': 'pointer', 'fontSize': '11px',
                           'fontFamily': 'monospace', 'marginRight': '4px', 'marginBottom': '4px'}
                ) for d in merged['date_str'].tolist()
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '4px'}),
            html.Div(id='sleep-detail-content', style={'marginTop': '16px'}),
        ]),
    ])

def page_daylighting():
    return html.Div([
        html.Div('Outdoor Solar → Indoor Brightness', style={'fontSize': '20px', 'fontWeight': '400', 'color': TEXT, 'marginBottom': '4px'}),
        html.P('Daily mean values, 08:00–18:00.', style={'fontSize': '12px', 'color': MUTED, 'marginBottom': '16px'}),
        card('', [dcc.Graph(figure=fig_s1, config={'displayModeBar': False}),
                  html.P(f'Outdoor solar explains {r1**2*100:.0f}% of indoor brightness variance (r={r1:+.3f}, p<0.001).',
                         style={'fontSize': '11px', 'color': MUTED, 'fontStyle': 'italic'})]),
        html.Div(style={'height': '1px', 'background': BORDER, 'margin': '4px 0 16px 0'}),
        html.Div('Indoor Brightness → Sleep Quality', style={'fontSize': '20px', 'fontWeight': '400', 'color': TEXT, 'marginBottom': '4px'}),
        html.P('Garmin labels sleep as date D+1, so daytime D precedes that night\'s sleep.', style={'fontSize': '12px', 'color': MUTED, 'marginBottom': '16px'}),
        card('', [dcc.Graph(figure=fig_s2, config={'displayModeBar': False}),
                  html.P(f'Brighter daytime → less deep sleep the following night (r={r2:+.3f}, p={p2:.3f} *).',
                         style={'fontSize': '11px', 'color': MUTED, 'fontStyle': 'italic'})]),
        card('', [dcc.Graph(figure=fig_s3, config={'displayModeBar': False}),
                  html.P(f'Higher daytime brightness → more variable respiration during sleep (r={r3:+.3f}, p={p3:.3f} *).',
                         style={'fontSize': '11px', 'color': MUTED, 'fontStyle': 'italic'})]),
    ])

def page_humidity():
    return html.Div([
        html.Div('Sleep-period Humidity → Sleep Quality', style={'fontSize': '20px', 'fontWeight': '400', 'color': TEXT, 'marginBottom': '4px'}),
        html.P('Cross-night analysis: each point = one night. n = 12 (Feb 28 excluded: sleep-period data unavailable).', style={'fontSize': '12px', 'color': MUTED, 'marginBottom': '16px'}),
        card('', [dcc.Graph(figure=fig_h1, config={'displayModeBar': False}),
                  html.P(f'Higher sleep-period humidity → lower sleep score (r={rh1:+.3f}, p={ph1:.3f} *).',
                         style={'fontSize': '11px', 'color': MUTED, 'fontStyle': 'italic'})]),
        card('', [dcc.Graph(figure=fig_h2, config={'displayModeBar': False}),
                  html.P(f'Higher humidity → elevated sleep HR (r={rh2:+.3f}, p={ph2:.3f} *).',
                         style={'fontSize': '11px', 'color': MUTED, 'fontStyle': 'italic'})]),
        card('', [dcc.Graph(figure=fig_h3, config={'displayModeBar': False}),
                  html.P(f'Higher humidity → faster respiration (r={rh3:+.3f}, p={ph3:.3f} *).',
                         style={'fontSize': '11px', 'color': MUTED, 'fontStyle': 'italic'})]),
        html.Div(style={'height': '1px', 'background': BORDER, 'margin': '4px 0 16px 0'}),
        html.Div('Within-night Analysis', style={'fontSize': '20px', 'fontWeight': '400', 'color': TEXT, 'marginBottom': '4px'}),
        html.P('Does humidity vary within a single night, and does it correlate with HR/resp changes?', style={'fontSize': '12px', 'color': MUTED, 'marginBottom': '16px'}),
        card('select night', [
            html.Div(
                id='night-btn-group',
                children=[
                    html.Button(
                        d, id=f'wbtn-{d}', n_clicks=0,
                        style={'padding': '5px 12px', 'borderRadius': '6px',
                               'border': f'1px solid {BORDER}', 'background': BG3,
                               'color': MUTED, 'cursor': 'pointer', 'fontSize': '11px',
                               'fontFamily': 'monospace', 'marginRight': '4px', 'marginBottom': '4px'}
                    ) for d in valid_nights['date_str'].tolist()
                ],
                style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '14px'}
            ),
            dcc.Store(id='night-selector', data=valid_nights['date_str'].tolist()[0]),
            dcc.Graph(id='within-night-chart', config={'displayModeBar': False}),
            html.Div(id='within-night-stats', style={'marginTop': '12px'}),
        ]),
        card('why cross-night but not within-night', [
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '16px', 'fontSize': '12px', 'color': MUTED}, children=[
                html.Div([html.Div('Cross-night (n=12) ✓', style={'color': GREEN, 'fontWeight': '500', 'marginBottom': '4px'}),
                          'Humidity range: 21–43% (Δ=22%) — large enough to drive sleep differences. Sleep score: r=−0.529, p=0.077 *']),
                html.Div([html.Div('Within-night (15-min) ✗', style={'color': RED, 'fontWeight': '500', 'marginBottom': '4px'}),
                          'Humidity range: 1–4% — too small, physiological noise dominates. Avg r=−0.125, no consistent direction.']),
            ]),
            html.P('Humidity acts as a cumulative environmental condition — the overall level matters, not moment-to-moment fluctuations.',
                   style={'fontSize': '11px', 'color': MUTED, 'fontStyle': 'italic', 'marginTop': '10px'}),
        ]),
        card('sleep-period humidity threshold explorer', [
            # slider row: label | track | value
            html.Div([
                html.Span('Humidity threshold',
                          style={'fontSize': '12px', 'color': MUTED, 'fontFamily': 'monospace',
                                 'whiteSpace': 'nowrap', 'marginRight': '16px', 'flexShrink': '0'}),
                dcc.Input(
                    id='hum-slider', type='range',
                    min=int(sub_hs['avg_humidity_sleep'].min()),
                    max=int(sub_hs['avg_humidity_sleep'].max()) + 1,
                    step=1,
                    value=int(sub_hs['avg_humidity_sleep'].median()),
                    style={'flex': '1', 'minWidth': '0'}),
                html.Span(id='hum-slider-label',
                          children=f"{int(sub_hs['avg_humidity_sleep'].median())}%",
                          style={'fontSize': '12px', 'color': BRIGHT, 'fontFamily': 'monospace',
                                 'fontWeight': '500', 'marginLeft': '16px', 'flexShrink': '0',
                                 'minWidth': '36px', 'textAlign': 'right'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '0',
                       'marginBottom': '16px'}),
            html.Div(id='hum-threshold-result'),
        ]),
    ])

@callback(Output('page-content', 'children'), Input('url', 'pathname'))
def render_page(pathname):
    if pathname == '/daylighting': return page_daylighting()
    if pathname == '/humidity':    return page_humidity()
    return page_overview()

# within-night button callbacks
from dash import ctx

@callback(
    Output('night-selector', 'data'),
    Output('night-btn-group', 'children'),
    [Input(f'wbtn-{d}', 'n_clicks') for d in valid_nights['date_str'].tolist()],
    prevent_initial_call=False,
)
def update_night_selector(*args):
    dates = valid_nights['date_str'].tolist()
    # determine selected: use ctx.triggered, default to first
    selected = dates[0]
    if ctx.triggered and ctx.triggered[0]['prop_id'] != '.':
        btn_id = ctx.triggered[0]['prop_id'].replace('.n_clicks', '')
        selected = btn_id.replace('wbtn-', '')
    buttons = [
        html.Button(
            d, id=f'wbtn-{d}', n_clicks=0,
            style={
                'padding': '5px 12px', 'borderRadius': '6px', 'cursor': 'pointer',
                'fontSize': '11px', 'fontFamily': 'monospace',
                'marginRight': '4px', 'marginBottom': '4px',
                'border': f'1px solid {"#a78bfa" if d == selected else BORDER}',
                'background': 'rgba(167,139,250,0.15)' if d == selected else BG3,
                'color': '#a78bfa' if d == selected else MUTED,
            }
        ) for d in dates
    ]
    return selected, buttons

@callback(
    Output('within-night-chart', 'figure'),
    Output('within-night-stats', 'children'),
    Input('night-selector', 'data')
)
def update_within_night(selected):
    night = valid_nights[valid_nights['date_str'] == selected].iloc[0]

    mask_in   = (indoor_raw['timestamp'] >= night['sleep_start_dt']) & \
                (indoor_raw['timestamp'] <= night['sleep_end_dt'])
    mask_hr   = hr_raw['ts'].between(night['sleep_start_dt'], night['sleep_end_dt'])
    mask_resp = resp_raw['ts'].between(night['sleep_start_dt'], night['sleep_end_dt'])

    indoor_n = indoor_raw[mask_in][['timestamp', 'humidity']].copy()
    indoor_n['ts_15'] = indoor_n['timestamp'].dt.round('15min')
    indoor_n = indoor_n.groupby('ts_15')['humidity'].mean().reset_index()

    hr_n = hr_raw[mask_hr].copy()
    hr_n['ts_15'] = hr_n['ts'].dt.round('15min')
    hr_15 = hr_n.groupby('ts_15')['hr_value'].mean().reset_index()

    resp_n = resp_raw[mask_resp].copy()
    resp_n['ts_15'] = resp_n['ts'].dt.round('15min')
    resp_15 = resp_n.groupby('ts_15')['resp_value'].mean().reset_index()

    m = indoor_n.merge(hr_15, on='ts_15').merge(resp_15, on='ts_15')

    if len(m) < 5:
        return go.Figure(), html.P('Insufficient data', style={'color': RED})

    r_hr,   p_hr   = stats.pearsonr(m['humidity'], m['hr_value'])
    r_resp, p_resp = stats.pearsonr(m['humidity'], m['resp_value'])
    hum_range = m['humidity'].max() - m['humidity'].min()

    # normalize resp to HR scale for visual — hover still shows real values
    resp_scaled = m['resp_value'] / m['resp_value'].mean() * m['hr_value'].mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=m['ts_15'], y=m['humidity'],
        name='Humidity (%)',
        line=dict(color=HUM, width=2, shape='spline', smoothing=1.0),
        fill='tozeroy', fillcolor='rgba(56,189,248,0.1)',
        hovertemplate='%{x|%H:%M}<br>Humidity: %{y:.1f}%<extra></extra>'
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=m['ts_15'], y=m['hr_value'],
        name='HR (bpm)',
        line=dict(color=RED, width=1.5, shape='spline', smoothing=1.0),
        hovertemplate='%{x|%H:%M}<br>HR: %{y:.1f} bpm<extra></extra>'
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=m['ts_15'], y=resp_scaled,
        name='Resp (br/min)',
        line=dict(color='#fb923c', width=1.5, dash='dot', shape='spline', smoothing=1.0),
        customdata=m['resp_value'],
        hovertemplate='%{x|%H:%M}<br>Resp: %{customdata:.1f} br/min<extra></extra>'
    ), secondary_y=True)

    fig.update_layout(**plot_layout(
        height=280, hovermode='x unified',
        legend=dict(orientation='h', y=1.12, font=dict(color=TEXT, size=11)),
        margin=dict(t=10, b=40, l=60, r=60),
        hoverlabel=dict(bgcolor=BG2, bordercolor=BORDER, font=dict(color=TEXT, size=11))
    ))
    fig.update_xaxes(tickformat='%H:%M', **NOGRID, tickfont=dict(color=TEXT))
    fig.update_yaxes(
        title_text='Humidity (%)', secondary_y=False,
        title_font=dict(color=HUM, size=10), tickfont=dict(color=MUTED), **GRID
    )
    fig.update_yaxes(
        title_text='HR (bpm)',
        secondary_y=True,
        title_font=dict(color=RED, size=10), tickfont=dict(color=MUTED), showgrid=False
    )

    stats_div = html.Div(
        style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '12px'},
        children=[
            html.Div([
                html.Div('Humidity range',
                         style={'fontSize': '10px', 'color': MUTED, 'textTransform': 'uppercase',
                                'letterSpacing': '.8px', 'fontFamily': 'monospace', 'marginBottom': '4px'}),
                html.Div(f'{hum_range:.0f}%',
                         style={'fontSize': '20px', 'fontWeight': '500', 'color': TEXT}),
                html.Div(f'{m["humidity"].min():.0f}–{m["humidity"].max():.0f}%',
                         style={'fontSize': '10px', 'color': MUTED}),
            ], style={'background': BG2, 'border': f'1px solid {BORDER}', 'borderRadius': '8px', 'padding': '12px'}),

            html.Div([
                html.Div('Humidity vs HR',
                         style={'fontSize': '10px', 'color': MUTED, 'textTransform': 'uppercase',
                                'letterSpacing': '.8px', 'fontFamily': 'monospace', 'marginBottom': '4px'}),
                html.Div(f'r = {r_hr:+.3f}',
                         style={'fontSize': '20px', 'fontWeight': '500', 'color': TEXT}),
                html.Div(f'p = {p_hr:.3f}',
                         style={'fontSize': '10px', 'color': MUTED}),
            ], style={'background': BG2, 'border': f'1px solid {BORDER}', 'borderRadius': '8px', 'padding': '12px'}),

            html.Div([
                html.Div('Humidity vs Resp',
                         style={'fontSize': '10px', 'color': MUTED, 'textTransform': 'uppercase',
                                'letterSpacing': '.8px', 'fontFamily': 'monospace', 'marginBottom': '4px'}),
                html.Div(f'r = {r_resp:+.3f}',
                         style={'fontSize': '20px', 'fontWeight': '500', 'color': TEXT}),
                html.Div(f'p = {p_resp:.3f}',
                         style={'fontSize': '10px', 'color': MUTED}),
            ], style={'background': BG2, 'border': f'1px solid {BORDER}', 'borderRadius': '8px', 'padding': '12px'}),
        ]
    )

    return fig, stats_div

@callback(Output('hum-threshold-result', 'children'), Input('hum-slider', 'value'))
def update_hum_threshold(threshold):
    threshold = float(threshold)
    above = merged[merged['avg_humidity_sleep'] > threshold].dropna(subset=['avg_humidity_sleep','sleep_score'])
    below = merged[merged['avg_humidity_sleep'] <= threshold].dropna(subset=['avg_humidity_sleep','sleep_score'])
    avg_a = f'{above["sleep_score"].mean():.1f}' if len(above) else '—'
    avg_b = f'{below["sleep_score"].mean():.1f}' if len(below) else '—'
    return html.Div([
        html.Div([html.Span(f'Nights with sleep-period humidity > {int(threshold)}%: '),
                  html.Span(f'{len(above)} nights', style={'fontWeight': '500', 'color': TEXT}),
                  html.Span(', avg sleep score '),
                  html.Span(avg_a, style={'fontWeight': '500', 'color': RED})],
                 style={'marginBottom': '6px', 'fontSize': '12px', 'color': MUTED}),
        html.Div([html.Span(f'Nights with sleep-period humidity ≤ {int(threshold)}%: '),
                  html.Span(f'{len(below)} nights', style={'fontWeight': '500', 'color': TEXT}),
                  html.Span(', avg sleep score '),
                  html.Span(avg_b, style={'fontWeight': '500', 'color': GREEN})],
                 style={'fontSize': '12px', 'color': MUTED}),
        html.P('Drag to explore: nights above threshold tend to have lower sleep scores (r = −0.529, p = 0.077 *)',
               style={'fontSize': '10px', 'color': MUTED, 'fontStyle': 'italic', 'marginTop': '8px'}),
    ], style={'background': BG, 'padding': '12px', 'borderRadius': '6px'})

@callback(Output('sleep-night-selector', 'value', allow_duplicate=True),
    Input('nbtn-2026-02-17', 'n_clicks'), prevent_initial_call=True)
def _set_night_2026_02_17(n): return '2026-02-17'

@callback(Output('sleep-night-selector', 'value', allow_duplicate=True),
    Input('nbtn-2026-02-18', 'n_clicks'), prevent_initial_call=True)
def _set_night_2026_02_18(n): return '2026-02-18'

@callback(Output('sleep-night-selector', 'value', allow_duplicate=True),
    Input('nbtn-2026-02-19', 'n_clicks'), prevent_initial_call=True)
def _set_night_2026_02_19(n): return '2026-02-19'

@callback(Output('sleep-night-selector', 'value', allow_duplicate=True),
    Input('nbtn-2026-02-20', 'n_clicks'), prevent_initial_call=True)
def _set_night_2026_02_20(n): return '2026-02-20'

@callback(Output('sleep-night-selector', 'value', allow_duplicate=True),
    Input('nbtn-2026-02-21', 'n_clicks'), prevent_initial_call=True)
def _set_night_2026_02_21(n): return '2026-02-21'

@callback(Output('sleep-night-selector', 'value', allow_duplicate=True),
    Input('nbtn-2026-02-22', 'n_clicks'), prevent_initial_call=True)
def _set_night_2026_02_22(n): return '2026-02-22'

@callback(Output('sleep-night-selector', 'value', allow_duplicate=True),
    Input('nbtn-2026-02-23', 'n_clicks'), prevent_initial_call=True)
def _set_night_2026_02_23(n): return '2026-02-23'

@callback(Output('sleep-night-selector', 'value', allow_duplicate=True),
    Input('nbtn-2026-02-24', 'n_clicks'), prevent_initial_call=True)
def _set_night_2026_02_24(n): return '2026-02-24'

@callback(Output('sleep-night-selector', 'value', allow_duplicate=True),
    Input('nbtn-2026-02-25', 'n_clicks'), prevent_initial_call=True)
def _set_night_2026_02_25(n): return '2026-02-25'

@callback(Output('sleep-night-selector', 'value', allow_duplicate=True),
    Input('nbtn-2026-02-26', 'n_clicks'), prevent_initial_call=True)
def _set_night_2026_02_26(n): return '2026-02-26'

@callback(Output('sleep-night-selector', 'value', allow_duplicate=True),
    Input('nbtn-2026-02-27', 'n_clicks'), prevent_initial_call=True)
def _set_night_2026_02_27(n): return '2026-02-27'

@callback(Output('sleep-night-selector', 'value', allow_duplicate=True),
    Input('nbtn-2026-02-28', 'n_clicks'), prevent_initial_call=True)
def _set_night_2026_02_28(n): return '2026-02-28'

@callback(Output('sleep-night-selector', 'value', allow_duplicate=True),
    Input('nbtn-2026-03-01', 'n_clicks'), prevent_initial_call=True)
def _set_night_2026_03_01(n): return '2026-03-01'

@callback(Output('sleep-detail-content', 'children'), Input('sleep-night-selector', 'value'))
def update_sleep_detail(selected):
    if not selected:
        selected = merged['date_str'].tolist()[0]
    row = merged[merged['date_str'] == selected].iloc[0]
    light_score = min(100, int(row['avg_brightness'] * 1.5 + row['avg_solar'] / 5))
    score = int(row['sleep_score'])
    awake_pct = row['awake_h'] / row['total_sleep_h'] * 100
    score_color = GREEN if score >= 80 else YELLOW if score >= 65 else RED
    ls_color = GREEN if light_score >= 70 else YELLOW if light_score >= 40 else RED
    hum_sleep = f"{row['avg_humidity_sleep']:.1f}%" if not pd.isna(row.get('avg_humidity_sleep', float('nan'))) else 'N/A'
    stacked_bar = html.Div([
        html.Div([
            html.Div(style={'display': 'flex', 'width': '100%', 'borderRadius': '4px',
                            'overflow': 'hidden', 'marginBottom': '6px'}, children=[
                html.Div(style={'width': f"{row['deep_pct']:.1f}%", 'height': '20px', 'background': DEEP}),
                html.Div(style={'width': f"{row['rem_pct']:.1f}%", 'height': '20px', 'background': REM}),
                html.Div(style={'width': f"{row['light_pct']:.1f}%", 'height': '20px', 'background': '#4b5563'}),
                html.Div(style={'width': f"{awake_pct:.1f}%", 'height': '20px', 'background': RED}),
            ]),
            html.Div([
                html.Span([html.Span(style={'display': 'inline-block', 'width': '8px', 'height': '8px',
                                           'background': DEEP, 'borderRadius': '2px', 'marginRight': '4px'}),
                           f"deep {row['deep_pct']:.0f}%"],
                          style={'fontSize': '10px', 'color': MUTED, 'fontFamily': 'monospace', 'marginRight': '14px'}),
                html.Span([html.Span(style={'display': 'inline-block', 'width': '8px', 'height': '8px',
                                           'background': REM, 'borderRadius': '2px', 'marginRight': '4px'}),
                           f"REM {row['rem_pct']:.0f}%"],
                          style={'fontSize': '10px', 'color': MUTED, 'fontFamily': 'monospace', 'marginRight': '14px'}),
                html.Span([html.Span(style={'display': 'inline-block', 'width': '8px', 'height': '8px',
                                           'background': '#4b5563', 'borderRadius': '2px', 'marginRight': '4px'}),
                           f"light {row['light_pct']:.0f}%"],
                          style={'fontSize': '10px', 'color': MUTED, 'fontFamily': 'monospace', 'marginRight': '14px'}),
                html.Span([html.Span(style={'display': 'inline-block', 'width': '8px', 'height': '8px',
                                           'background': RED, 'borderRadius': '2px', 'marginRight': '4px'}),
                           f"awake {awake_pct:.0f}%"],
                          style={'fontSize': '10px', 'color': MUTED, 'fontFamily': 'monospace'}),
            ]),
        ]),
    ], style={'marginBottom': '16px'})
    return html.Div([
        html.Div(f'Sleep night {selected} · Daytime {row["light_date"]}',
                 style={'fontSize': '10px', 'color': MUTED, 'fontFamily': 'monospace',
                        'letterSpacing': '.8px', 'textTransform': 'uppercase', 'marginBottom': '12px'}),
        stacked_bar,
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginTop': '14px'}, children=[
            html.Div([
                html.Div('Sleep metrics', style={'fontSize': '10px', 'color': MUTED, 'textTransform': 'uppercase',
                                                 'letterSpacing': '.8px', 'fontFamily': 'monospace', 'marginBottom': '10px'}),
                stat_row('Sleep score', badge(score, score)),
                stat_row('Total sleep', f'{row["total_sleep_h"]:.1f} h'),
                stat_row('Deep / REM / Light', f'{row["deep_pct"]:.1f}% / {row["rem_pct"]:.1f}% / {row["light_pct"]:.1f}%'),
                stat_row('Mean sleep HR', f'{row["hr_mean"]:.0f} bpm'),
                stat_row('Resting HR', f'{int(row["resting_hr"])} bpm'),
                stat_row('Resp rate', f'{row["resp_mean"]:.1f} br/min'),
                stat_row('Body battery gain', f'+{int(row["body_battery_change"])}'),
            ]),
            html.Div([
                html.Div(f'Daytime conditions ({row["light_date"][5:]})',
                         style={'fontSize': '10px', 'color': MUTED, 'textTransform': 'uppercase',
                                'letterSpacing': '.8px', 'fontFamily': 'monospace', 'marginBottom': '10px'}),
                stat_row('Lighting score', html.Span(f'{light_score}/100',
                         style={'color': ls_color, 'fontWeight': '500', 'fontFamily': 'monospace'})),
                stat_row('Solar radiation', f'{row["avg_solar"]:.0f} W/m²'),
                stat_row('Indoor brightness', f'{row["avg_brightness"]:.1f}%'),
                stat_row('Cloud cover', f'{row["avg_cloud"]:.0f}%'),
                stat_row('Pre-sleep humidity', f'{row["avg_humidity_presleep"]:.1f}%'),
                stat_row('Sleep-period humidity', hum_sleep),
                stat_row('Indoor temp', f'{row["avg_temp"]:.1f} °C'),
            ]),
        ]),
    ])

# clientside callback — update range track fill + label
app.clientside_callback(
    '''
    function(value) {
        var slider = document.getElementById('hum-slider');
        if (slider) {
            var mn = parseFloat(slider.min);
            var mx = parseFloat(slider.max);
            var pct = (parseFloat(value) - mn) / (mx - mn) * 100;
            slider.style.background = 'linear-gradient(to right, #a78bfa 0%, #a78bfa ' + pct + '%, #242428 ' + pct + '%, #242428 100%)';
        }
        return parseInt(value) + '%';
    }
    ''',
    Output('hum-slider-label', 'children'),
    Input('hum-slider', 'value'),
    prevent_initial_call=False,
)

server = app.server  # expose for gunicorn

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)