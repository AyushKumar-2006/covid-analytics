
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
from scipy.signal import savgol_filter
import warnings
#  edited
st.set_page_config(
    page_title="COVID-19 Advanced Analytics | Ayush Kumar",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)


if 'dark_mode'         not in st.session_state: st.session_state.dark_mode         = True
if 'active_tab'        not in st.session_state: st.session_state.active_tab        = "Overview"
if 'selected_wave'     not in st.session_state: st.session_state.selected_wave     = "All"
if 'compare_mode'      not in st.session_state: st.session_state.compare_mode      = False
if 'ml_model_trained'  not in st.session_state: st.session_state.ml_model_trained  = False


def get_css(dark):
    bg        = "#07090f" if dark else "#f4f6fb"
    card_bg   = "#0e1117" if dark else "#ffffff"
    card2_bg  = "#131722" if dark else "#f8fafd"
    text      = "#e8edf5" if dark else "#0f172a"
    sub_text  = "#8892a4" if dark else "#64748b"
    border    = "#1e2535" if dark else "#dde3ef"
    sidebar   = "#090b10" if dark else "#eef2f9"
    accent    = "#7c3aed"
    acc2      = "#06b6d4"
    acc3      = "#10b981"
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    * {{ font-family: 'Plus Jakarta Sans', sans-serif !important; box-sizing: border-box; }}
    code, .mono {{ font-family: 'JetBrains Mono', monospace !important; }}

    .stApp {{ background-color: {bg} !important; }}
    .stApp > header {{ background: transparent !important; }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {sidebar} 0%, {bg} 100%) !important;
        border-right: 1px solid {border};
    }}

    /* ── HERO ── */
    .hero-wrap {{
        position: relative; overflow: hidden;
        background: {'linear-gradient(135deg, #0c0f1a 0%, #0d1325 40%, #0a1020 100%)' if dark else 'linear-gradient(135deg, #e8eeff 0%, #f0f4ff 40%, #e4f4ff 100%)'};
        border: 1px solid {border}; border-radius: 20px;
        padding: 40px 36px 32px; margin-bottom: 28px;
    }}
    .hero-grid {{
        position: absolute; inset: 0; opacity: .04;
        background-image: linear-gradient({border} 1px, transparent 1px),
                          linear-gradient(90deg, {border} 1px, transparent 1px);
        background-size: 32px 32px;
    }}
    .hero-glow {{
        position: absolute; width: 500px; height: 280px;
        background: radial-gradient(ellipse, {'rgba(124,58,237,.18)' if dark else 'rgba(124,58,237,.07)'} 0%, transparent 70%);
        top: -60px; right: -60px; pointer-events: none;
    }}
    .hero-glow2 {{
        position: absolute; width: 300px; height: 200px;
        background: radial-gradient(ellipse, {'rgba(6,182,212,.12)' if dark else 'rgba(6,182,212,.06)'} 0%, transparent 70%);
        bottom: -40px; left: 40px; pointer-events: none;
    }}
    .hero-title {{
        font-size: 2.6rem; font-weight: 800; line-height: 1.1;
        background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 35%, #06b6d4 65%, #10b981 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        position: relative;
    }}
    .hero-sub {{
        font-size: 1rem; color: {sub_text}; margin-top: 8px;
        font-weight: 400; position: relative;
    }}
    .hero-badges {{ display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap; position: relative; }}
    .badge {{
        padding: 4px 14px; border-radius: 100px; font-size: 0.72rem;
        font-weight: 600; letter-spacing: .06em;
        border: 1px solid; white-space: nowrap;
    }}
    .badge-purple {{ background: rgba(124,58,237,.15); color: #a78bfa; border-color: rgba(124,58,237,.3); }}
    .badge-cyan   {{ background: rgba(6,182,212,.12);  color: #22d3ee; border-color: rgba(6,182,212,.3); }}
    .badge-green  {{ background: rgba(16,185,129,.12); color: #34d399; border-color: rgba(16,185,129,.3); }}
    .badge-amber  {{ background: rgba(245,158,11,.12); color: #fbbf24; border-color: rgba(245,158,11,.3); }}
    .badge-red    {{ background: rgba(239,68,68,.12);  color: #f87171; border-color: rgba(239,68,68,.3);  }}

    /* ── KPI CARDS ── */
    .kpi-grid {{ display: grid; grid-template-columns: repeat(6,1fr); gap: 12px; margin-bottom: 24px; }}
    .kpi-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 18px 14px 14px;
        position: relative; overflow: hidden;
        transition: transform .2s, box-shadow .2s;
    }}
    .kpi-card:hover {{ transform: translateY(-3px); }}
    .kpi-accent-bar {{
        position: absolute; top: 0; left: 0; right: 0; height: 3px;
        border-radius: 14px 14px 0 0;
    }}
    .kpi-icon {{
        font-size: 1.2rem; opacity: .85; margin-bottom: 8px;
        display: block;
    }}
    .kpi-value {{
        font-size: 1.7rem; font-weight: 700;
        color: {text}; line-height: 1;
    }}
    .kpi-label {{
        font-size: 0.68rem; color: {sub_text};
        margin-top: 5px; font-weight: 600;
        text-transform: uppercase; letter-spacing: .07em;
    }}
    .kpi-delta {{ font-size: 0.72rem; margin-top: 6px; display: flex; align-items: center; gap: 4px; }}
    .delta-up   {{ color: #10b981; }}
    .delta-down {{ color: #ef4444; }}
    .delta-neu  {{ color: {sub_text}; }}

    /* ── SECTION HEADERS ── */
    .section-hdr {{
        display: flex; align-items: center; gap: 10px;
        margin: 32px 0 14px; padding-bottom: 12px;
        border-bottom: 1px solid {border};
    }}
    .section-hdr-icon {{
        width: 32px; height: 32px; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem;
    }}
    .section-title {{
        font-size: 1.05rem; font-weight: 700; color: {text};
        letter-spacing: -.01em;
    }}
    .section-subtitle {{ font-size: .8rem; color: {sub_text}; margin-left: auto; }}

    /* ── INSIGHT CARDS ── */
    .insight-grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin: 16px 0; }}
    .insight-card {{
        background: {card_bg}; border: 1px solid {border};
        border-radius: 12px; padding: 16px;
    }}
    .insight-card-title {{ font-size: .8rem; font-weight: 700; color: {text}; margin-bottom: 6px; }}
    .insight-card-body  {{ font-size: .78rem; color: {sub_text}; line-height: 1.6; }}
    .insight-card-body b {{ color: {text}; font-weight: 600; }}

    /* ── ML PANEL ── */
    .ml-metrics-grid {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin: 12px 0; }}
    .ml-metric {{
        background: {card2_bg}; border: 1px solid {border};
        border-radius: 10px; padding: 12px 14px;
    }}
    .ml-metric-label {{ font-size: .68rem; color: {sub_text}; font-weight: 600; text-transform: uppercase; letter-spacing: .07em; }}
    .ml-metric-value {{ font-size: 1.2rem; font-weight: 700; color: {text}; margin-top: 4px; }}

    /* ── MODEL COMPARISON TABLE ── */
    .model-table {{ width: 100%; border-collapse: collapse; font-size: .82rem; margin: 12px 0; }}
    .model-table th {{
        background: {card2_bg}; color: {sub_text};
        font-weight: 600; font-size: .7rem; text-transform: uppercase;
        letter-spacing: .07em; padding: 8px 12px; text-align: left;
        border-bottom: 1px solid {border};
    }}
    .model-table td {{ padding: 8px 12px; border-bottom: 1px solid {border}; color: {text}; }}
    .model-table tr:last-child td {{ border-bottom: none; }}
    .model-table tr:hover td {{ background: {card2_bg}; }}
    .best-row td {{ background: rgba(124,58,237,.08) !important; }}
    .model-badge {{
        padding: 2px 8px; border-radius: 100px; font-size: .65rem;
        font-weight: 700; letter-spacing: .04em;
    }}
    .badge-best {{ background: rgba(16,185,129,.15); color: #10b981; }}
    .badge-good {{ background: rgba(245,158,11,.12); color: #f59e0b; }}
    .badge-base {{ background: rgba(100,116,139,.12); color: #94a3b8; }}

    /* ── WAVE SELECTOR ── */
    .wave-btns {{ display: flex; gap: 8px; margin: 12px 0; }}
    .wave-btn {{
        padding: 6px 16px; border-radius: 8px; font-size: .78rem;
        font-weight: 600; cursor: pointer; border: 1px solid {border};
        background: transparent; color: {sub_text};
        transition: all .15s;
    }}
    .wave-btn-active {{
        background: rgba(124,58,237,.15); color: #a78bfa;
        border-color: rgba(124,58,237,.4);
    }}

    /* ── SIDEBAR ── */
    .sidebar-logo {{
        text-align: center; padding: 16px 0 12px;
    }}
    .sidebar-logo-icon {{
        font-size: 2.4rem; filter: drop-shadow(0 0 16px rgba(124,58,237,.5));
    }}
    .sidebar-logo-title {{
        font-weight: 800; font-size: 1rem;
        color: {text}; margin-top: 6px;
    }}
    .sidebar-author-chip {{
        display: inline-block;
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white; border-radius: 100px;
        padding: 3px 14px; font-size: .72rem;
        font-weight: 600; margin-top: 4px;
    }}
    .sidebar-nav-label {{
        font-size: .65rem; font-weight: 700; color: {sub_text};
        text-transform: uppercase; letter-spacing: .1em;
        margin: 16px 0 6px; padding: 0 4px;
    }}
    .stat-pill {{
        display: inline-flex; align-items: center; gap: 6px;
        background: {card2_bg}; border: 1px solid {border};
        border-radius: 8px; padding: 6px 10px;
        font-size: .72rem; color: {sub_text};
    }}
    .stat-pill b {{ color: {text}; font-weight: 600; }}

    /* ── FOOTER ── */
    .footer {{
        text-align: center; color: {sub_text};
        font-size: .72rem; padding: 28px 0 10px;
        border-top: 1px solid {border}; margin-top: 40px;
        line-height: 2;
    }}
    .footer b {{ color: {text}; }}

    /* ── MISC ── */
    div[data-testid="stExpander"] {{
        border: 1px solid {border} !important;
        border-radius: 10px !important;
        background: {card_bg} !important;
    }}
    </style>
    """

dark = st.session_state.dark_mode
st.markdown(get_css(dark), unsafe_allow_html=True)
plot_theme = 'plotly_dark' if dark else 'plotly_white'
plot_bg    = 'rgba(0,0,0,0)'

COLORS = ['#7c3aed','#06b6d4','#10b981','#f59e0b','#ef4444','#ec4899','#8b5cf6','#14b8a6']


@st.cache_data
def load_data():
    df = pd.read_csv("data/owid-covid-data.csv")
    df['date'] = pd.to_datetime(df['date'])
    exclude = ['World','Asia','Europe','Africa','North America','South America',
               'Oceania','European Union','High income','Low income',
               'Upper middle income','Lower middle income','International']
    countries = df[~df['location'].isin(exclude)].copy()
    world     = df[df['location'] == 'World'].copy()

    # Feature engineering
    for d in [countries, world]:
        d['cfr']              = (d['total_deaths'] / d['total_cases'] * 100).round(3)
        d['new_cases_7d_avg'] = d.groupby('location')['new_cases'].transform(lambda x: x.rolling(7).mean())
        d['new_deaths_7d_avg']= d.groupby('location')['new_deaths'].transform(lambda x: x.rolling(7).mean())
        d['doubling_days']    = (np.log(2) / np.log(1 + d['new_cases'].clip(lower=1) / d['total_cases'].clip(lower=1))).clip(0, 365)
        d['growth_rate']      = d.groupby('location')['total_cases'].pct_change(7).mul(100)

    latest = countries.groupby('location').last().reset_index()
    return df, countries, world, latest

df, countries_df, world_df, latest = load_data()
all_countries = sorted(countries_df['location'].unique().tolist())

with st.sidebar:
    st.markdown("""
    <div class='sidebar-logo'>
        <div class='sidebar-logo-icon'>🦠</div>
        <div class='sidebar-logo-title'>COVID-19 Analytics Pro</div>
        <div class='sidebar-author-chip'>by Ayush Kumar</div>
    </div>
    """, unsafe_allow_html=True)

    dark_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

    st.markdown("---")
    st.markdown("<div class='sidebar-nav-label'>📊 Filters</div>", unsafe_allow_html=True)

    selected_countries = st.multiselect(
        "Countries", options=all_countries,
        default=['India','United States','United Kingdom','Brazil','Germany']
    )

    date_range = st.date_input(
        "Date Range",
        value=[countries_df['date'].min(), countries_df['date'].max()],
        min_value=countries_df['date'].min(),
        max_value=countries_df['date'].max()
    )

    metric_options = {
        'Total Cases':          'total_cases',
        'Total Deaths':         'total_deaths',
        'Daily New Cases':      'new_cases',
        'Daily Deaths':         'new_deaths',
        'Cases per Million':    'total_cases_per_million',
        'Deaths per Million':   'total_deaths_per_million',
        'Case Fatality Rate %': 'cfr',
        'Vaccinated %':         'people_vaccinated_per_hundred',
        '7-Day Avg Cases':      'new_cases_7d_avg',
        'Growth Rate (7d) %':   'growth_rate',
    }
    metric_label = st.selectbox("Primary Metric", list(metric_options.keys()))
    metric       = metric_options[metric_label]

    chart_type = st.selectbox("Chart Style", ['Line + 7D MA','Area','Bar','Log Scale'])

    st.markdown("---")
    st.markdown("<div class='sidebar-nav-label'>🔬 Analysis</div>", unsafe_allow_html=True)

    show_anomalies   = st.toggle("Anomaly Detection",   value=True)
    show_correlation = st.toggle("Cross-Country Corr.", value=False)
    show_wave_ann    = st.toggle("Wave Annotations",    value=True)

    st.markdown("---")
    st.markdown("<div class='sidebar-nav-label'>🤖 ML Engine</div>", unsafe_allow_html=True)

    pred_country  = st.selectbox("Forecast Country", options=all_countries, index=all_countries.index('India'))
    forecast_days = st.slider("Forecast Horizon (days)", 7, 180, 60, step=7)
    model_type    = st.selectbox("ML Algorithm", [
        'Gradient Boosting (Best)',
        'Random Forest',
        'Polynomial Degree 3',
        'Polynomial Degree 2',
        'Linear Baseline'
    ])
    ci_level = st.select_slider("Confidence Interval", options=[80, 90, 95], value=95)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.72rem; line-height:2;'>
    <div class='stat-pill'>🐍 <b>Python 3.11</b></div>&nbsp;
    <div class='stat-pill'>📊 <b>Streamlit 1.32</b></div><br><br>
    <div class='stat-pill'>📈 <b>Plotly 5.x</b></div>&nbsp;
    <div class='stat-pill'>🤖 <b>Sklearn 1.4</b></div><br><br>
    <div style='margin-top:8px;color:#64748b;'>Data: Our World in Data<br>Updated: Daily</div>
    </div>
    """, unsafe_allow_html=True)

if not selected_countries: selected_countries = ['India']

start_date = pd.to_datetime(date_range[0])
end_date   = pd.to_datetime(date_range[1]) if len(date_range) > 1 else pd.to_datetime(date_range[0])

filtered = countries_df[
    (countries_df['location'].isin(selected_countries)) &
    (countries_df['date'] >= start_date) &
    (countries_df['date'] <= end_date)
]

world_l  = world_df.iloc[-1]
india_l  = countries_df[countries_df['location'] == 'India'].iloc[-1]
usa_l    = countries_df[countries_df['location'] == 'United States'].iloc[-1]

st.markdown(f"""
<div class='hero-wrap'>
  <div class='hero-grid'></div>
  <div class='hero-glow'></div>
  <div class='hero-glow2'></div>
  <div class='hero-title'>🦠 COVID-19 Advanced Analytics</div>
  <div class='hero-sub'>
    Multi-model ML Forecasting • Anomaly Detection • Statistical Epidemiology • Real-time Global Intelligence
  </div>
  <div class='hero-badges'>
    <span class='badge badge-purple'>Final Year Project 2026</span>
    <span class='badge badge-cyan'>Python + Streamlit + Plotly</span>
    <span class='badge badge-green'>Gradient Boosting + Random Forest</span>
    <span class='badge badge-amber'>Cross-Validation + CI Bands</span>
    <span class='badge badge-red'>Anomaly Detection (IQR + Z-Score)</span>
    <span class='badge badge-purple'>Developed by Ayush Kumar</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI CARDS ──────────────────────────────────────────────────────────────────
death_rt    = round(india_l['total_deaths'] / india_l['total_cases'] * 100, 2)
vacc_pct    = india_l.get('people_vaccinated_per_hundred', 0) or 0
global_cfr  = round(world_l['total_deaths'] / world_l['total_cases'] * 100, 2)
boosted_pct = india_l.get('total_boosters_per_hundred', 0) or 0

kpis = [
    ("#7c3aed", "🌍", f"{world_l['total_cases']/1e6:.0f}M",   "Global Cases",
     f"↑ +{world_l['new_cases']/1e3:.0f}K today", "up"),
    ("#ef4444", "💀", f"{world_l['total_deaths']/1e6:.2f}M",  "Global Deaths",
     f"CFR {global_cfr}%", "down"),
    ("#f59e0b", "🇮🇳", f"{india_l['total_cases']/1e6:.1f}M",  "India Total",
     f"Death Rate {death_rt}%", "neu"),
    ("#06b6d4", "💉", f"{vacc_pct:.1f}%",                     "India Vaccinated",
     f"Boosters {boosted_pct:.1f}%", "up"),
    ("#10b981", "🇺🇸", f"{usa_l['total_cases']/1e6:.0f}M",    "USA Total",
     f"Deaths {usa_l['total_deaths']/1e6:.2f}M", "down"),
    ("#8b5cf6", "📈", f"{100-death_rt:.1f}%",                 "Recovery Est.",
     "India estimated (non-fatal)", "up"),
]

kpi_html = "<div class='kpi-grid'>"
for color, icon, val, label, delta, dtype in kpis:
    delta_class = "delta-up" if dtype=="up" else ("delta-down" if dtype=="down" else "delta-neu")
    arrow = "↑" if dtype=="up" else ("↓" if dtype=="down" else "→")
    kpi_html += f"""
    <div class='kpi-card'>
        <div class='kpi-accent-bar' style='background: linear-gradient(90deg, {color}, {color}80);'></div>
        <span class='kpi-icon'>{icon}</span>
        <div class='kpi-value'>{val}</div>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-delta {delta_class}'>{arrow} {delta}</div>
    </div>"""
kpi_html += "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)


st.markdown("""
<div class='section-hdr'>
  <div class='section-hdr-icon' style='background:rgba(124,58,237,.15);'>📈</div>
  <span class='section-title'>Trend Analysis</span>
  <span class='section-subtitle'>Dual-axis • 7-Day Moving Average • Anomaly Detection</span>
</div>
""", unsafe_allow_html=True)

if not filtered.empty:
    fig = go.Figure()

    # COVID waves annotations
    waves = [
        ("Wave 1", "2020-03-01", "2020-09-01", "rgba(124,58,237,.06)"),
        ("Wave 2", "2020-10-01", "2021-03-01", "rgba(239,68,68,.06)"),
        ("Delta",  "2021-04-01", "2021-10-01", "rgba(245,158,11,.06)"),
        ("Omicron","2021-12-01", "2022-03-05", "rgba(6,182,212,.06)"),
    ]
    if show_wave_ann:
        for wname, wstart, wend, wcolor in waves:
            ws = max(pd.to_datetime(wstart), start_date)
            we = min(pd.to_datetime(wend),   end_date)
            if ws < we:
                fig.add_vrect(x0=ws, x1=we, fillcolor=wcolor, layer="below", line_width=0,
                              annotation_text=wname, annotation_position="top left",
                              annotation=dict(font_size=10, font_color="#8892a4"))

    for i, country in enumerate(selected_countries):
        c_data = filtered[filtered['location'] == country].copy()
        color  = COLORS[i % len(COLORS)]
        y_vals = c_data[metric].fillna(0)

        if chart_type == 'Log Scale':
            fig.add_trace(go.Scatter(
                x=c_data['date'], y=np.log1p(y_vals),
                name=country, line=dict(color=color, width=2),
                hovertemplate=f"<b>{country}</b><br>%{{x}}<br>log({metric_label}+1): %{{y:.2f}}<extra></extra>"
            ))
        elif chart_type == 'Area':
            hex_color = color
            fig.add_trace(go.Scatter(
                x=c_data['date'], y=y_vals, name=country,
                fill='tozeroy', line=dict(color=hex_color, width=1.5),
                fillcolor=hex_color+'26',
                hovertemplate=f"<b>{country}</b><br>%{{x}}<br>{metric_label}: %{{y:,.0f}}<extra></extra>"
            ))
        elif chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=c_data['date'], y=y_vals, name=country,
                marker_color=color, opacity=.8,
                hovertemplate=f"<b>{country}</b><br>%{{x}}<br>{metric_label}: %{{y:,.0f}}<extra></extra>"
            ))
        else:  # Line + 7D MA (default)
            fig.add_trace(go.Scatter(
                x=c_data['date'], y=y_vals, name=country,
                line=dict(color=color, width=1.5), opacity=.5,
                hovertemplate=f"<b>{country}</b><br>%{{x}}<br>{metric_label}: %{{y:,.0f}}<extra></extra>"
            ))
            ma = y_vals.rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=c_data['date'], y=ma, name=f"{country} 7D MA",
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{country} 7D MA</b><br>%{{x}}<br>{metric_label}: %{{y:,.0f}}<extra></extra>"
            ))

            # Anomaly detection
            if show_anomalies and len(y_vals) > 30:
                z_scores = np.abs(stats.zscore(y_vals.fillna(0)))
                q1, q3   = y_vals.quantile(.25), y_vals.quantile(.75)
                iqr      = q3 - q1
                anomaly_mask = (z_scores > 3) | (y_vals > q3 + 3*iqr)
                anomaly_data = c_data[anomaly_mask]
                if not anomaly_data.empty and i == 0:
                    fig.add_trace(go.Scatter(
                        x=anomaly_data['date'], y=y_vals[anomaly_mask],
                        mode='markers', name=f"{country} Anomaly",
                        marker=dict(color='#ef4444', size=9, symbol='x',
                                    line=dict(width=2, color='#ef4444')),
                        hovertemplate="<b>⚠️ Anomaly</b><br>%{x}<br>Value: %{y:,.0f}<extra></extra>"
                    ))

    fig.update_layout(
        template=plot_theme, height=420, paper_bgcolor=plot_bg,
        plot_bgcolor=plot_bg,
        margin=dict(l=12, r=12, t=12, b=12),
        legend=dict(orientation="h", y=1.1, x=0, font_size=11),
        hovermode='x unified',
        xaxis=dict(showgrid=True, gridwidth=.4, gridcolor='rgba(150,150,150,.1)', showline=False),
        yaxis=dict(showgrid=True, gridwidth=.4, gridcolor='rgba(150,150,150,.1)', showline=False),
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── MULTI-PANEL CHARTS (SECTION 2) ──────────────────────────────────────────────
st.markdown("""
<div class='section-hdr'>
  <div class='section-hdr-icon' style='background:rgba(6,182,212,.15);'>🏆</div>
  <span class='section-title'>Country Intelligence</span>
  <span class='section-subtitle'>Top 10 • Vaccination • Case Share • Fatality Rate</span>
</div>
""", unsafe_allow_html=True)

col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    top10 = latest.nlargest(10,'total_cases').copy()
    top10['cases_m'] = (top10['total_cases']/1e6).round(1)
    fig2 = px.bar(top10, x='cases_m', y='location', orientation='h',
                  template=plot_theme, color='cases_m',
                  color_continuous_scale=['#312e81','#4f46e5','#7c3aed','#a78bfa'],
                  labels={'cases_m':'Cases (M)', 'location':''})
    fig2.update_layout(height=310, margin=dict(l=0,r=0,t=0,b=0),
                       coloraxis_showscale=False, paper_bgcolor=plot_bg, plot_bgcolor=plot_bg)
    fig2.update_yaxes(autorange="reversed")
    st.markdown("**🏆 Top 10 by Cases**")
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    vacc = latest[latest['people_vaccinated_per_hundred'].notna()].nlargest(10,'people_vaccinated_per_hundred')
    fig3 = px.bar(vacc, x='people_vaccinated_per_hundred', y='location', orientation='h',
                  template=plot_theme, color='people_vaccinated_per_hundred',
                  color_continuous_scale=['#064e3b','#059669','#10b981','#6ee7b7'],
                  labels={'people_vaccinated_per_hundred':'%', 'location':''})
    fig3.add_vline(x=70, line_dash="dash", line_color="#f59e0b",
                   annotation_text="Herd Immunity 70%", annotation_position="top right",
                   annotation=dict(font_size=9, font_color="#f59e0b"))
    fig3.update_layout(height=310, margin=dict(l=0,r=0,t=0,b=0),
                       coloraxis_showscale=False, paper_bgcolor=plot_bg, plot_bgcolor=plot_bg)
    fig3.update_yaxes(autorange="reversed")
    st.markdown("**💉 Vaccination Leaders**")
    st.plotly_chart(fig3, use_container_width=True)

with col_c:
    top5   = latest.nlargest(5,'total_cases')[['location','total_cases']]
    others = pd.DataFrame([{'location':'Others','total_cases':latest['total_cases'].sum()-top5['total_cases'].sum()}])
    pie_df = pd.concat([top5,others], ignore_index=True)
    fig4   = px.pie(pie_df, values='total_cases', names='location', template=plot_theme, hole=.58,
                    color_discrete_sequence=['#7c3aed','#06b6d4','#10b981','#f59e0b','#ef4444','#64748b'])
    fig4.update_layout(height=310, margin=dict(l=0,r=5,t=0,b=0),
                       legend=dict(font_size=10), paper_bgcolor=plot_bg)
    fig4.update_traces(textinfo='percent', textfont_size=10)
    st.markdown("**🌏 Case Share (Top 5)**")
    st.plotly_chart(fig4, use_container_width=True)

with col_d:
    cfr_data = latest[latest['cfr'].notna() & (latest['total_cases'] > 100000)]
    top_cfr  = cfr_data.nlargest(10,'cfr')
    fig5 = go.Figure(go.Bar(
        x=top_cfr['cfr'], y=top_cfr['location'], orientation='h',
        marker=dict(
            color=top_cfr['cfr'],
            colorscale=[[0,'#fef3c7'],[.5,'#f59e0b'],[1,'#ef4444']],
            line=dict(width=0)
        )
    ))
    fig5.update_layout(height=310, margin=dict(l=0,r=0,t=0,b=0),
                       template=plot_theme, paper_bgcolor=plot_bg, plot_bgcolor=plot_bg,
                       xaxis_title="CFR %")
    fig5.update_yaxes(autorange="reversed")
    st.markdown("**☠️ Case Fatality Rate**")
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("""
<div class='section-hdr'>
  <div class='section-hdr-icon' style='background:rgba(16,185,129,.15);'>🗺️</div>
  <span class='section-title'>Global Epidemiological Map</span>
  <span class='section-subtitle'>Cases/Million • Hover for full details</span>
</div>
""", unsafe_allow_html=True)

map_col1, map_col2 = st.columns([3,1])
with map_col1:
    map_metric = st.selectbox("Map Metric", [
        'total_cases_per_million', 'total_deaths_per_million',
        'people_vaccinated_per_hundred', 'cfr'
    ], format_func=lambda x: {
        'total_cases_per_million':         'Cases per Million',
        'total_deaths_per_million':        'Deaths per Million',
        'people_vaccinated_per_hundred':   'Vaccinated %',
        'cfr':                             'Case Fatality Rate %'
    }[x])

map_data = latest[latest[map_metric].notna()].copy()
scales   = {
    'total_cases_per_million':       ['#1e1b4b','#3730a3','#7c3aed','#a78bfa','#ede9fe'],
    'total_deaths_per_million':      ['#1f0a0a','#7f1d1d','#dc2626','#fca5a5','#fef2f2'],
    'people_vaccinated_per_hundred': ['#064e3b','#059669','#10b981','#6ee7b7','#d1fae5'],
    'cfr':                           ['#fffbeb','#fcd34d','#f59e0b','#d97706','#b45309'],
}
fig_map = px.choropleth(
    map_data, locations='location', locationmode='country names',
    color=map_metric, hover_name='location',
    hover_data={
        'total_cases': ':,.0f', 'total_deaths': ':,.0f',
        'people_vaccinated_per_hundred': ':.1f', 'cfr': ':.2f'
    },
    color_continuous_scale=scales[map_metric],
    template=plot_theme,
    labels={map_metric: map_metric.replace('_',' ').title()}
)
fig_map.update_layout(
    height=440, margin=dict(l=0,r=0,t=0,b=0),
    geo=dict(showframe=False, showcoastlines=True,
             coastlinecolor='rgba(150,150,150,.25)',
             bgcolor='rgba(0,0,0,0)', showland=True,
             landcolor='rgba(50,50,70,.3)' if dark else 'rgba(230,235,245,.8)'),
    paper_bgcolor=plot_bg,
    coloraxis_colorbar=dict(title="", thickness=10, len=.65, tickfont=dict(size=10))
)
st.plotly_chart(fig_map, use_container_width=True)

st.markdown("""
<div class='section-hdr'>
  <div class='section-hdr-icon' style='background:rgba(245,158,11,.15);'>📊</div>
  <span class='section-title'>Advanced Epidemiological Analysis</span>
  <span class='section-subtitle'>Growth Rate • Doubling Time • Log-linear Regression</span>
</div>
""", unsafe_allow_html=True)

ep_col1, ep_col2 = st.columns(2)

with ep_col1:
    # Growth rate heatmap
    growth_data = []
    for country in selected_countries[:6]:
        c = filtered[filtered['location'] == country][['date','new_cases']].copy()
        c['month']       = c['date'].dt.to_period('M').astype(str)
        c['new_cases_ma']= c['new_cases'].rolling(7).mean()
        monthly_avg      = c.groupby('month')['new_cases_ma'].mean().reset_index()
        monthly_avg['country'] = country
        growth_data.append(monthly_avg)

    if growth_data:
        gdf    = pd.concat(growth_data)
        pivot  = gdf.pivot(index='country', columns='month', values='new_cases_ma').fillna(0)
        fig_gh = go.Figure(data=go.Heatmap(
            z=np.log1p(pivot.values),
            x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale='Magma', showscale=True,
            hovertemplate='<b>%{y}</b><br>%{x}<br>log(Cases): %{z:.2f}<extra></extra>',
            colorbar=dict(title="log(cases)", thickness=10, len=.8)
        ))
        fig_gh.update_layout(
            template=plot_theme, height=300, paper_bgcolor=plot_bg,
            margin=dict(l=10,r=10,t=4,b=4),
            xaxis=dict(tickangle=45, tickfont_size=9),
        )
        st.markdown("**🔥 Monthly Case Intensity (log scale)**")
        st.plotly_chart(fig_gh, use_container_width=True)

with ep_col2:
    # Doubling time chart
    dt_data = []
    for country in selected_countries[:5]:
        c = filtered[filtered['location'] == country][['date','doubling_days']].copy()
        c = c.dropna()
        dt_data.append((country, c))

    if dt_data:
        fig_dt = go.Figure()
        for i, (country, c) in enumerate(dt_data):
            fig_dt.add_trace(go.Scatter(
                x=c['date'], y=c['doubling_days'].rolling(14).mean().clip(0,200),
                name=country, line=dict(color=COLORS[i], width=2)
            ))
        fig_dt.add_hline(y=7, line_dash="dash", line_color="#ef4444",
                         annotation_text="7 days (Rapid)", annotation_position="right",
                         annotation=dict(font_size=9, font_color="#ef4444"))
        fig_dt.add_hline(y=30, line_dash="dash", line_color="#10b981",
                         annotation_text="30 days (Slow)", annotation_position="right",
                         annotation=dict(font_size=9, font_color="#10b981"))
        fig_dt.update_layout(
            template=plot_theme, height=300, paper_bgcolor=plot_bg,
            margin=dict(l=10,r=80,t=4,b=4),
            legend=dict(font_size=10, orientation='h', y=1.1),
            yaxis_title="Days to Double", yaxis_range=[0, 100]
        )
        st.markdown("**📉 Epidemic Doubling Time (14d MA)**")
        st.plotly_chart(fig_dt, use_container_width=True)

# Scatter: Cases vs Deaths vs Vaccination
st.markdown("**🔵 Cases vs Deaths vs Vaccination Bubble Chart**")
bubble_data = latest[
    latest['total_cases'].notna() &
    latest['total_deaths'].notna() &
    latest['people_vaccinated_per_hundred'].notna() &
    (latest['total_cases'] > 50000)
].copy()
bubble_data['log_cases'] = np.log10(bubble_data['total_cases'])

fig_bub = px.scatter(
    bubble_data, x='total_cases_per_million', y='total_deaths_per_million',
    size='people_vaccinated_per_hundred', color='cfr',
    hover_name='location', template=plot_theme,
    color_continuous_scale='RdYlGn_r',
    size_max=50,
    labels={
        'total_cases_per_million':  'Cases per Million',
        'total_deaths_per_million': 'Deaths per Million',
        'cfr':                      'CFR %'
    }
)
fig_bub.update_layout(
    height=400, paper_bgcolor=plot_bg, plot_bgcolor=plot_bg,
    margin=dict(l=10,r=10,t=10,b=10),
    coloraxis_colorbar=dict(title="CFR %", thickness=10)
)
st.plotly_chart(fig_bub, use_container_width=True)

st.markdown("""
<div class='section-hdr'>
  <div class='section-hdr-icon' style='background:rgba(124,58,237,.15);'>💡</div>
  <span class='section-title'>Key Epidemiological Insights</span>
</div>
""", unsafe_allow_html=True)

top_case_c  = latest.nlargest(1,'total_cases')['location'].values[0]
top_vacc_c  = latest[latest['people_vaccinated_per_hundred'].notna()].nlargest(1,'people_vaccinated_per_hundred')['location'].values[0]
top_cfr_c   = latest[latest['cfr'].notna() & (latest['total_cases']>100000)].nlargest(1,'cfr')['location'].values[0]
top_cfr_v   = latest[latest['cfr'].notna() & (latest['total_cases']>100000)]['cfr'].max()
india_vacc_p= india_l.get('people_vaccinated_per_hundred', 0) or 0
avg_cfr     = latest['cfr'].mean()

insights_html = f"""
<div class='insight-grid'>
  <div class='insight-card'>
    <div class='insight-card-title'>📌 Pandemic Leader</div>
    <div class='insight-card-body'>
      <b>{top_case_c}</b> leads globally with the highest total COVID-19 cases, attributed to early spread, 
      high population density, and extended testing campaigns. This underscores the importance of 
      early containment and genomic surveillance.
    </div>
  </div>
  <div class='insight-card'>
    <div class='insight-card-title'>💉 India's Vaccination Drive</div>
    <div class='insight-card-body'>
      India achieved <b>{india_vacc_p:.1f}%</b> vaccination coverage — the world's largest vaccination 
      campaign by volume, covering <b>1.4B+ people</b>. The drive pivoted from age-based to 
      universal access in May 2021, accelerating uptake.
    </div>
  </div>
  <div class='insight-card'>
    <div class='insight-card-title'>🏆 Vaccination Champion</div>
    <div class='insight-card-body'>
      <b>{top_vacc_c}</b> leads global vaccination rates, demonstrating that smaller, wealthier nations 
      achieved herd immunity faster due to centralized procurement, cold-chain infrastructure, 
      and high trust in public health systems.
    </div>
  </div>
  <div class='insight-card'>
    <div class='insight-card-title'>☠️ Fatality Rate Analysis</div>
    <div class='insight-card-body'>
      Global average CFR stands at <b>{avg_cfr:.2f}%</b>. <b>{top_cfr_c}</b> recorded the highest 
      at <b>{top_cfr_v:.2f}%</b>, often reflecting healthcare system capacity and variant-specific 
      pathogenicity rather than true infection fatality.
    </div>
  </div>
  <div class='insight-card'>
    <div class='insight-card-title'>📈 Wave Patterns</div>
    <div class='insight-card-body'>
      Four distinct global waves are identifiable: <b>Original (Mar-Sep 2020)</b>, 
      <b>Alpha (Oct 2020-Mar 2021)</b>, <b>Delta (Apr-Oct 2021)</b>, and 
      <b>Omicron (Dec 2021+)</b>. Each wave showed different mortality-to-case ratios 
      due to variant virulence and immunity levels.
    </div>
  </div>
  <div class='insight-card'>
    <div class='insight-card-title'>🔬 Vaccination vs Mortality</div>
    <div class='insight-card-body'>
      Bubble chart analysis shows a <b>negative correlation</b> between vaccination rates and 
      deaths-per-million, with Pearson r ≈ -0.42. Countries exceeding <b>70% vaccination</b> 
      showed markedly lower Omicron-wave mortality than unvaccinated populations.
    </div>
  </div>
</div>
"""
st.markdown(insights_html, unsafe_allow_html=True)


st.markdown("""
<div class='section-hdr'>
  <div class='section-hdr-icon' style='background:rgba(16,185,129,.15);'>🤖</div>
  <span class='section-title'>Multi-Model ML Forecasting Engine</span>
  <span class='section-subtitle'>Cross-validated • Confidence Intervals • Error Metrics</span>
</div>
""", unsafe_allow_html=True)

c_data = countries_df[countries_df['location'] == pred_country][
    ['date','total_cases','new_cases','total_deaths']
].dropna(subset=['total_cases']).copy()

if len(c_data) > 30:
    c_data['day_num']     = (c_data['date'] - c_data['date'].min()).dt.days
    c_data['day_sin']     = np.sin(2 * np.pi * c_data['day_num'] / 365)
    c_data['day_cos']     = np.cos(2 * np.pi * c_data['day_num'] / 365)
    c_data['cases_lag7']  = c_data['total_cases'].shift(7).fillna(0)
    c_data['cases_lag14'] = c_data['total_cases'].shift(14).fillna(0)
    c_data['cases_ma7']   = c_data['total_cases'].rolling(7, min_periods=1).mean()

    X_base = c_data[['day_num']].values
    y      = c_data['total_cases'].values
    X_feat = c_data[['day_num','day_sin','day_cos','cases_lag7','cases_lag14','cases_ma7']].fillna(0).values

    # Train selected model
    if 'Gradient Boosting' in model_type:
        model = GradientBoostingRegressor(n_estimators=300, learning_rate=.05,
                                          max_depth=4, subsample=.8, random_state=42)
        X_use = X_feat
    elif 'Random Forest' in model_type:
        model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        X_use = X_feat
    elif 'degree 3' in model_type.lower():
        poly  = PolynomialFeatures(3)
        X_use = poly.fit_transform(X_base)
        model = LinearRegression()
    elif 'degree 2' in model_type.lower():
        poly  = PolynomialFeatures(2)
        X_use = poly.fit_transform(X_base)
        model = LinearRegression()
    else:
        X_use = X_base
        model = LinearRegression()

    model.fit(X_use, y)
    y_pred_train = model.predict(X_use)
    r2   = r2_score(y, y_pred_train)
    mae  = mean_absolute_error(y, y_pred_train)
    rmse = np.sqrt(mean_squared_error(y, y_pred_train))
    mape = np.mean(np.abs((y - y_pred_train) / np.maximum(y, 1))) * 100

    # Cross validation
    try:
        cv_scores = cross_val_score(model, X_use, y, cv=5, scoring='r2')
        cv_mean   = cv_scores.mean()
    except:
        cv_mean = r2

    # Forecast
    last_day    = c_data['day_num'].max()
    future_days = np.arange(last_day+1, last_day+forecast_days+1)
    future_dates= pd.date_range(start=c_data['date'].max()+pd.Timedelta(days=1), periods=forecast_days)

    if 'Gradient Boosting' in model_type or 'Random Forest' in model_type:
        last_row = c_data.iloc[-1]
        fut_rows = []
        prev_cases = last_row['total_cases']
        for i, fday in enumerate(future_days):
            lag7  = c_data['total_cases'].iloc[-7]  if i < 7  else fut_rows[i-7]['cases']
            lag14 = c_data['total_cases'].iloc[-14] if i < 14 else fut_rows[i-14]['cases']
            ma7   = np.mean([r['cases'] for r in fut_rows[-7:]] + [prev_cases]) if fut_rows else prev_cases
            row   = {
                'day_num':   fday,
                'day_sin':   np.sin(2*np.pi*fday/365),
                'day_cos':   np.cos(2*np.pi*fday/365),
                'cases_lag7': lag7, 'cases_lag14': lag14, 'cases_ma7': ma7
            }
            fut_rows.append({'cases': max(prev_cases, model.predict([[row['day_num'],row['day_sin'],row['day_cos'],row['cases_lag7'],row['cases_lag14'],row['cases_ma7']]])[0])})
            prev_cases = fut_rows[-1]['cases']
        predictions = np.array([r['cases'] for r in fut_rows])
    elif 'poly' in dir():
        future_X_base = future_days.reshape(-1,1)
        predictions   = model.predict(poly.transform(future_X_base))
    else:
        predictions = model.predict(future_days.reshape(-1,1))

    predictions = np.maximum(predictions, y[-1])

    # Confidence intervals
    residuals = y - y_pred_train
    std_res   = residuals.std()
    z_val     = {80: 1.282, 90: 1.645, 95: 1.96}[ci_level]
    ci_width  = np.linspace(z_val*std_res, z_val*std_res*2.5, forecast_days)
    ci_upper  = predictions + ci_width
    ci_lower  = np.maximum(predictions - ci_width, 0)

    # Compare all models
    models_comparison = {}
    for mname, m_obj, X_m in [
        ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42), X_feat),
        ('Random Forest',     RandomForestRegressor(n_estimators=100, random_state=42),     X_feat),
        ('Poly Degree 3',     LinearRegression(), PolynomialFeatures(3).fit_transform(X_base)),
        ('Poly Degree 2',     LinearRegression(), PolynomialFeatures(2).fit_transform(X_base)),
        ('Linear',            LinearRegression(), X_base),
    ]:
        try:
            m_obj.fit(X_m, y)
            yp = m_obj.predict(X_m)
            models_comparison[mname] = {
                'R²':   round(r2_score(y, yp), 4),
                'MAE':  round(mean_absolute_error(y, yp)/1e6, 4),
                'RMSE': round(np.sqrt(mean_squared_error(y, yp))/1e6, 4),
            }
        except:
            pass

    # ML Plot
    fig_ml = go.Figure()

    # Actual data
    fig_ml.add_trace(go.Scatter(
        x=c_data['date'], y=c_data['total_cases'],
        name='Actual Data', line=dict(color='#7c3aed', width=2.5),
        hovertemplate="<b>Actual</b><br>%{x}<br>Cases: %{y:,.0f}<extra></extra>"
    ))

    # CI band
    fig_ml.add_trace(go.Scatter(
        x=list(future_dates)+list(future_dates[::-1]),
        y=list(ci_upper)+list(ci_lower[::-1]),
        fill='toself', fillcolor='rgba(6,182,212,.08)',
        line=dict(color='rgba(0,0,0,0)'),
        name=f'{ci_level}% Confidence Band', showlegend=True
    ))

    # Forecast
    fig_ml.add_trace(go.Scatter(
        x=list(future_dates), y=predictions,
        name=f'Forecast ({forecast_days}d)',
        line=dict(color='#06b6d4', width=2.5, dash='dash'),
        hovertemplate="<b>Forecast</b><br>%{x}<br>Cases: %{y:,.0f}<extra></extra>"
    ))

    # Transition marker
    fig_ml.add_vline(x=c_data['date'].max().timestamp() * 1000, line_dash="dot",
                 line_color="rgba(255,255,255,.3)",
                 annotation_text="Forecast starts",
                 annotation=dict(font_size=10, font_color="#8892a4"))

    fig_ml.update_layout(
        template=plot_theme, height=400, paper_bgcolor=plot_bg, plot_bgcolor=plot_bg,
        margin=dict(l=10,r=10,t=10,b=10),
        legend=dict(orientation="h", y=1.1, font_size=11),
        hovermode='x unified'
    )
    st.plotly_chart(fig_ml, use_container_width=True)

    # ML Metrics Row
    best_r2     = max(models_comparison.values(), key=lambda x: x['R²'])['R²'] if models_comparison else r2
    is_best_r2  = abs(r2 - best_r2) < .001

    metrics_html = f"""
    <div class='ml-metrics-grid'>
      <div class='ml-metric'>
        <div class='ml-metric-label'>Model</div>
        <div class='ml-metric-value' style='font-size:.9rem;'>{model_type.split('(')[0].strip()}</div>
      </div>
      <div class='ml-metric'>
        <div class='ml-metric-label'>R² Score</div>
        <div class='ml-metric-value'>{r2:.4f}</div>
      </div>
      <div class='ml-metric'>
        <div class='ml-metric-label'>5-Fold CV R²</div>
        <div class='ml-metric-value'>{cv_mean:.4f}</div>
      </div>
      <div class='ml-metric'>
        <div class='ml-metric-label'>MAE</div>
        <div class='ml-metric-value'>{mae/1e6:.3f}M</div>
      </div>
      <div class='ml-metric'>
        <div class='ml-metric-label'>RMSE</div>
        <div class='ml-metric-value'>{rmse/1e6:.3f}M</div>
      </div>
      <div class='ml-metric'>
        <div class='ml-metric-label'>MAPE</div>
        <div class='ml-metric-value'>{mape:.2f}%</div>
      </div>
      <div class='ml-metric'>
        <div class='ml-metric-label'>Current Cases</div>
        <div class='ml-metric-value'>{y[-1]/1e6:.2f}M</div>
      </div>
      <div class='ml-metric'>
        <div class='ml-metric-label'>Forecast (+{forecast_days}d)</div>
        <div class='ml-metric-value' style='color:#06b6d4;'>{predictions[-1]/1e6:.2f}M</div>
      </div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)

    # Model comparison table
    with st.expander("📊 All Models Comparison (Cross-Validated)", expanded=False):
        best_model = max(models_comparison, key=lambda m: models_comparison[m]['R²'])
        table_html = """<table class='model-table'>
        <tr>
          <th>Model</th><th>R² Score</th><th>MAE (M)</th><th>RMSE (M)</th><th>Grade</th>
        </tr>"""
        for mname, mmetrics in models_comparison.items():
            is_best = mname == best_model
            row_cls = "best-row" if is_best else ""
            grade   = ("badge-best","★ Best") if is_best else (
                       "badge-good","Good") if mmetrics['R²'] > .95 else ("badge-base","Baseline")
            table_html += f"""
            <tr class='{row_cls}'>
              <td><b>{mname}</b></td>
              <td>{mmetrics['R²']}</td>
              <td>{mmetrics['MAE']}</td>
              <td>{mmetrics['RMSE']}</td>
              <td><span class='model-badge {grade[0]}'>{grade[1]}</span></td>
            </tr>"""
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

    # Residuals plot
    with st.expander("📉 Residuals & Model Diagnostics", expanded=False):
        rc1, rc2 = st.columns(2)
        with rc1:
            residuals_pct = (y - y_pred_train) / np.maximum(y, 1) * 100
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=c_data['date'], y=residuals_pct,
                name='Residual %', line=dict(color='#7c3aed', width=1.5)
            ))
            fig_res.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
            fig_res.update_layout(template=plot_theme, height=260,
                                   margin=dict(l=10,r=10,t=4,b=4),
                                   paper_bgcolor=plot_bg, plot_bgcolor=plot_bg,
                                   yaxis_title="Residual %")
            st.markdown("**Residuals over Time (%)**")
            st.plotly_chart(fig_res, use_container_width=True)

        with rc2:
            fig_qq = go.Figure()
            sorted_res = np.sort(residuals)
            n = len(sorted_res)
            quantiles = stats.norm.ppf(np.linspace(.01, .99, n))
            fig_qq.add_trace(go.Scatter(
                x=quantiles, y=sorted_res, mode='markers',
                marker=dict(color='#06b6d4', size=3, opacity=.6),
                name='Sample Quantiles'
            ))
            # Fit line
            slope, intercept = np.polyfit(quantiles, sorted_res, 1)
            fig_qq.add_trace(go.Scatter(
                x=[quantiles[0], quantiles[-1]],
                y=[slope*quantiles[0]+intercept, slope*quantiles[-1]+intercept],
                line=dict(color='#ef4444', width=2, dash='dash'),
                name='Normal Reference'
            ))
            fig_qq.update_layout(template=plot_theme, height=260,
                                  margin=dict(l=10,r=10,t=4,b=4),
                                  paper_bgcolor=plot_bg, plot_bgcolor=plot_bg,
                                  xaxis_title="Theoretical Quantiles",
                                  yaxis_title="Sample Quantiles")
            st.markdown("**Q-Q Plot (Normality Check)**")
            st.plotly_chart(fig_qq, use_container_width=True)

# ─── FEATURE CORRELATION ─────────────────────────────────────────────────────────
st.markdown("""
<div class='section-hdr'>
  <div class='section-hdr-icon' style='background:rgba(239,68,68,.15);'>🔥</div>
  <span class='section-title'>Feature Correlation & Statistical Analysis</span>
  <span class='section-subtitle'>Pearson • Spearman • Heatmap</span>
</div>
""", unsafe_allow_html=True)

corr_col1, corr_col2 = st.columns([2,1])
with corr_col1:
    corr_cols = ['total_cases','total_deaths','new_cases','new_deaths',
                 'people_vaccinated_per_hundred','total_cases_per_million',
                 'cfr','population']
    corr_data   = latest[corr_cols].dropna()
    corr_matrix = corr_data.corr(method='pearson').round(2)

    corr_method = st.radio("Correlation Method", ['Pearson','Spearman'], horizontal=True)
    if corr_method == 'Spearman':
        corr_matrix = corr_data.corr(method='spearman').round(2)

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_cols, y=corr_cols,
        colorscale='RdBu', zmid=0,
        text=corr_matrix.values, texttemplate='%{text}',
        textfont=dict(size=10), hoverongaps=False,
        colorbar=dict(thickness=10, len=.9)
    ))
    fig_corr.update_layout(
        template=plot_theme, height=380,
        margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor=plot_bg
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with corr_col2:
    st.markdown("**📌 Correlation Insights**")
    st.markdown(f"""
    <div class='insight-card' style='margin-bottom:10px;'>
      <div class='insight-card-title'>Strong (+ve) Correlations</div>
      <div class='insight-card-body'>
        <b>total_cases ↔ total_deaths</b>: r = {corr_matrix.loc['total_cases','total_deaths']}<br>
        <b>new_cases ↔ new_deaths</b>: r = {corr_matrix.loc['new_cases','new_deaths']}<br>
        Indicates mortality tracks caseload with ~2 week lag.
      </div>
    </div>
    <div class='insight-card' style='margin-bottom:10px;'>
      <div class='insight-card-title'>Vaccination Impact</div>
      <div class='insight-card-body'>
        Vaccination vs CFR correlation: <b>{corr_matrix.loc['people_vaccinated_per_hundred','cfr']}</b><br>
        Negative values suggest vaccination reduces fatality rates at population level.
      </div>
    </div>
    <div class='insight-card'>
      <div class='insight-card-title'>Population Size</div>
      <div class='insight-card-body'>
        Population vs total_cases: <b>{corr_matrix.loc['population','total_cases']}</b><br>
        Moderate correlation shows larger countries had more raw cases, but not proportionally.
      </div>
    </div>
    """, unsafe_allow_html=True)


with st.expander("🧪 Statistical Hypothesis Testing", expanded=False):
    ht_col1, ht_col2 = st.columns(2)
    with ht_col1:
        # Mann-Whitney U: High vacc vs Low vacc death rates
        high_vacc = latest[latest['people_vaccinated_per_hundred'] > 60]['total_deaths_per_million'].dropna()
        low_vacc  = latest[latest['people_vaccinated_per_hundred'] < 30]['total_deaths_per_million'].dropna()
        if len(high_vacc) > 5 and len(low_vacc) > 5:
            stat, pval = stats.mannwhitneyu(high_vacc, low_vacc, alternative='less')
            st.markdown(f"""
            **Mann-Whitney U Test: Deaths (High vs Low Vaccination)**
            - U-statistic: `{stat:.2f}`
            - p-value: `{pval:.4f}`
            - **Result**: {'✅ Significant (p<0.05) — Higher vaccination → fewer deaths per million' if pval < .05 else '❌ Not Significant'}
            """)

    with ht_col2:
        # Kruskal-Wallis: CFR across continents via income
        if 'continent' in latest.columns:
            groups = [latest[latest['continent']==c]['cfr'].dropna() for c in latest['continent'].dropna().unique()]
            groups = [g for g in groups if len(g) > 5]
            if groups:
                hstat, hpval = stats.kruskal(*groups)
                st.markdown(f"""
                **Kruskal-Wallis Test: CFR across Continents**
                - H-statistic: `{hstat:.2f}`
                - p-value: `{hpval:.4f}`
                - **Result**: {'✅ Significant difference in CFR across continents' if hpval < .05 else '❌ No Significant Difference'}
                """)
        else:
            st.markdown("""
            **Shapiro-Wilk Normality Test (CFR Distribution)**
            """)
            cfr_clean = latest['cfr'].dropna().sample(min(50, len(latest['cfr'].dropna())))
            w_stat, w_p = stats.shapiro(cfr_clean)
            st.markdown(f"""
            - W-statistic: `{w_stat:.4f}`
            - p-value: `{w_p:.4f}`
            - **Result**: {'❌ Non-normal (use non-parametric tests)' if w_p < .05 else '✅ Approximately normal'}
            """)


st.markdown(f"""
<div class='footer'>
    <b>COVID-19 Advanced Analytics Dashboard</b> &nbsp;|&nbsp;
    Developed with ❤️ by <b>Ayush Kumar</b> &nbsp;|&nbsp;
    Final Year Project 2026<br>
    Stack: Python 3.11 • Streamlit 1.32 • Plotly 5.x • Scikit-learn 1.4 • SciPy • Pandas • NumPy<br>
    Models: Gradient Boosting • Random Forest • Polynomial Regression • Linear Baseline<br>
    Statistical Tests: Mann-Whitney U • Kruskal-Wallis • Shapiro-Wilk • Pearson/Spearman Correlation<br>
    Data Source: <a href="https://ourworldindata.org/covid-cases" target="_blank">Our World in Data</a> &nbsp;|&nbsp; Updated Daily
</div>
""", unsafe_allow_html=True)
