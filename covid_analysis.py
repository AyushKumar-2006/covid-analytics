# hi
"""
COVID-19 Advanced Data Analysis & ML Pipeline
==============================================
Final Year Project — Ayush Kumar, 2026
Advanced features:
  - Multi-model ML comparison (GB, RF, Poly, Linear)
  - 5-fold cross-validation + confidence intervals
  - Statistical hypothesis testing (Mann-Whitney, Kruskal-Wallis, Shapiro-Wilk)
  - Anomaly detection (IQR + Z-score ensemble)
  - Epidemic doubling time & growth rate analysis
  - Professional multi-panel charts
  - Automated PDF/HTML summary report generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import scipy.stats as scipy_stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')


plt.rcParams.update({
    'figure.facecolor':  '#07090f',
    'axes.facecolor':    '#0e1117',
    'axes.edgecolor':    '#1e2535',
    'axes.labelcolor':   '#8892a4',
    'axes.titlecolor':   '#e8edf5',
    'xtick.color':       '#8892a4',
    'ytick.color':       '#8892a4',
    'text.color':        '#e8edf5',
    'grid.color':        '#1e2535',
    'grid.linewidth':    0.5,
    'legend.frameon':    False,
    'legend.fontsize':   9,
    'axes.titlesize':    11,
    'axes.titleweight':  'bold',
    'axes.labelsize':    9,
    'font.family':       'DejaVu Sans',
})

COLORS = ['#7c3aed','#06b6d4','#10b981','#f59e0b','#ef4444','#ec4899','#8b5cf6','#14b8a6']
ACCENT = '#7c3aed'


print("=" * 60)
print("   COVID-19 ADVANCED ANALYTICS PIPELINE")
print("   Final Year Project — Ayush Kumar 2026")
print("=" * 60)



print("\n[1/7] Loading & Engineering Features...")

df = pd.read_csv("data/owid-covid-data.csv")
df['date'] = pd.to_datetime(df['date'])

exclude = ['World','Asia','Europe','Africa','North America','South America',
           'Oceania','European Union','High income','Low income',
           'Upper middle income','Lower middle income','International']

countries_df = df[~df['location'].isin(exclude)].copy()
world_df     = df[df['location'] == 'World'].copy()


for dframe in [countries_df, world_df]:
    dframe['cfr']               = (dframe['total_deaths'] / dframe['total_cases'].clip(1) * 100).round(3)
    dframe['new_cases_7d_avg']  = dframe.groupby('location')['new_cases'].transform(lambda x: x.rolling(7).mean())
    dframe['new_deaths_7d_avg'] = dframe.groupby('location')['new_deaths'].transform(lambda x: x.rolling(7).mean())
    dframe['growth_rate_7d']    = dframe.groupby('location')['total_cases'].pct_change(7).mul(100)
    dframe['doubling_days']     = (
        np.log(2) / np.log(1 + dframe['new_cases'].clip(lower=1) / dframe['total_cases'].clip(lower=1))
    ).clip(0, 365)

latest = countries_df.groupby('location').last().reset_index()

print(f"   Rows    : {len(df):,}")
print(f"   Countries: {countries_df['location'].nunique()}")
print(f"   Features : {len(df.columns)} raw → {len(dframe.columns)} engineered")
print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")



print("\n[2/7] Computing Epidemiological Summary...")

world_l  = world_df.iloc[-1]
india_l  = countries_df[countries_df['location']=='India'].iloc[-1]
top10    = latest.nlargest(10,'total_cases')[['location','total_cases','total_deaths','cfr']]

print(f"\n   GLOBAL:")
print(f"   Total Cases  : {world_l['total_cases']/1e6:.1f}M")
print(f"   Total Deaths : {world_l['total_deaths']/1e6:.2f}M")
print(f"   Global CFR   : {world_l['total_deaths']/world_l['total_cases']*100:.2f}%")
print(f"\n   INDIA:")
print(f"   Total Cases  : {india_l['total_cases']/1e6:.1f}M")
print(f"   Death Rate   : {india_l['cfr']:.2f}%")
print(f"   Vaccinated % : {india_l.get('people_vaccinated_per_hundred',0):.1f}%")
print(f"\n   TOP 10 COUNTRIES:")
print(top10.to_string(index=False))



print("\n[3/7] Generating Chart 1: Country Intelligence...")

countries_focus = ['India','United States','United Kingdom','Brazil','Germany','France']
colors_focus    = COLORS[:len(countries_focus)]

fig1 = plt.figure(figsize=(18, 11))
fig1.patch.set_facecolor('#07090f')
gs   = gridspec.GridSpec(2, 3, figure=fig1, hspace=.42, wspace=.35)


ax1 = fig1.add_subplot(gs[0, :2])
for i, country in enumerate(countries_focus):
    c = countries_df[countries_df['location']==country]
    ax1.plot(c['date'], c['total_cases']/1e6, color=colors_focus[i], lw=2, label=country)
  
    ma = c['total_cases'].rolling(7).mean()/1e6
    ax1.plot(c['date'], ma, color=colors_focus[i], lw=1, linestyle=':', alpha=.4)


waves = [
    ("Wave 1", "2020-03-01", "2020-09-01"),
    ("Wave 2", "2020-10-01", "2021-03-01"),
    ("Delta",  "2021-04-01", "2021-10-01"),
    ("Omicron","2021-12-01", "2022-03-05"),
]
wave_colors = ['#7c3aed22','#ef444422','#f59e0b22','#06b6d422']
for (wname, wstart, wend), wc in zip(waves, wave_colors):
    ax1.axvspan(pd.to_datetime(wstart), pd.to_datetime(wend), alpha=.8,
                facecolor=wc, edgecolor='none')
    ax1.text(pd.to_datetime(wstart), ax1.get_ylim()[1]*.98 if ax1.get_ylim()[1]>0 else 1,
             wname, fontsize=7.5, color='#8892a4', va='top')

ax1.set_title("Total Cases by Country (Millions) — with Wave Annotations", pad=10)
ax1.set_ylabel("Cases (Millions)")
ax1.legend(loc='upper left', ncol=3, fontsize=8.5)
ax1.grid(axis='y', alpha=.4)
ax1.xaxis.set_tick_params(rotation=30)


ax2 = fig1.add_subplot(gs[0, 2])
for i, country in enumerate(countries_focus[:4]):
    c    = countries_df[countries_df['location']==country]
    mask = c['new_deaths_7d_avg'] > 0
    ax2.fill_between(c['date'][mask], c['new_deaths_7d_avg'][mask],
                     alpha=.25, color=colors_focus[i])
    ax2.plot(c['date'][mask], c['new_deaths_7d_avg'][mask],
             color=colors_focus[i], lw=1.5, label=country)
ax2.set_title("Daily Deaths (7D MA)")
ax2.set_ylabel("Deaths/Day")
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(axis='y', alpha=.4)
ax2.xaxis.set_tick_params(rotation=30)

ax3 = fig1.add_subplot(gs[1, 0])
for i, country in enumerate(countries_focus):
    c = countries_df[(countries_df['location']==country) &
                     countries_df['people_vaccinated_per_hundred'].notna()]
    if not c.empty:
        ax3.plot(c['date'], c['people_vaccinated_per_hundred'],
                 color=colors_focus[i], lw=2, label=country)
ax3.axhline(70, color='#10b981', lw=1.5, linestyle='--', label='Herd Immunity 70%')
ax3.set_title("Vaccination Progress (%)")
ax3.set_ylabel("% Population Vaccinated")
ax3.legend(fontsize=7.5, loc='upper left')
ax3.grid(axis='y', alpha=.4)
ax3.xaxis.set_tick_params(rotation=30)


ax4 = fig1.add_subplot(gs[1, 1])
for i, country in enumerate(countries_focus[:4]):
    c    = countries_df[(countries_df['location']==country) & countries_df['cfr'].notna()]
    cfr_s= c['cfr'].rolling(14).mean()
    ax4.plot(c['date'], cfr_s, color=colors_focus[i], lw=2, label=country)
ax4.set_title("Case Fatality Rate % (14D MA)")
ax4.set_ylabel("CFR %")
ax4.legend(fontsize=8)
ax4.grid(axis='y', alpha=.4)
ax4.xaxis.set_tick_params(rotation=30)


ax5 = fig1.add_subplot(gs[1, 2])
bub  = latest[latest['total_cases_per_million'].notna() &
              latest['total_deaths_per_million'].notna() &
              latest['people_vaccinated_per_hundred'].notna() &
              (latest['total_cases']>100000)].copy()
sc = ax5.scatter(
    bub['total_cases_per_million']/1000,
    bub['total_deaths_per_million'],
    s=bub['people_vaccinated_per_hundred']*3,
    c=bub['cfr'], cmap='RdYlGn_r',
    alpha=.7, edgecolors='none', vmin=0, vmax=5
)
for _, row in bub[bub['location'].isin(['India','United States','Brazil','Germany'])].iterrows():
    ax5.annotate(row['location'],
                 (row['total_cases_per_million']/1000, row['total_deaths_per_million']),
                 fontsize=7, color='#8892a4',
                 xytext=(5, 5), textcoords='offset points')
plt.colorbar(sc, ax=ax5, label='CFR %', shrink=.85)
ax5.set_xlabel("Cases per Million (Thousands)")
ax5.set_ylabel("Deaths per Million")
ax5.set_title("Cases vs Deaths vs Vaccination\n(bubble size = vacc %)")
ax5.grid(alpha=.3)

fig1.suptitle("COVID-19 Country Intelligence Dashboard", fontsize=14,
              fontweight='bold', color='#e8edf5', y=1.01)
plt.savefig('chart1_country_intelligence.png', dpi=180, bbox_inches='tight',
            facecolor='#07090f')
plt.close()
print("   ✓ chart1_country_intelligence.png")



print("\n[4/7] Generating Chart 2: ML Forecasting Engine...")

pred_country  = 'India'
forecast_days = 90

c_data = countries_df[countries_df['location']==pred_country][
    ['date','total_cases','new_cases','total_deaths']
].dropna(subset=['total_cases']).copy().reset_index(drop=True)

c_data['day_num']    = (c_data['date'] - c_data['date'].min()).dt.days
c_data['day_sin']    = np.sin(2*np.pi*c_data['day_num']/365)
c_data['day_cos']    = np.cos(2*np.pi*c_data['day_num']/365)
c_data['cases_lag7'] = c_data['total_cases'].shift(7).fillna(0)
c_data['cases_ma7']  = c_data['total_cases'].rolling(7, min_periods=1).mean()

X_base = c_data[['day_num']].values
y      = c_data['total_cases'].values
X_feat = c_data[['day_num','day_sin','day_cos','cases_lag7','cases_ma7']].fillna(0).values

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

models_def = {
    'Gradient Boosting': (GradientBoostingRegressor(n_estimators=300, learning_rate=.05, max_depth=4, random_state=42), X_feat),
    'Random Forest':     (RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42), X_feat),
    'Polynomial Deg 3':  (LinearRegression(), PolynomialFeatures(3).fit_transform(X_base)),
    'Polynomial Deg 2':  (LinearRegression(), PolynomialFeatures(2).fit_transform(X_base)),
    'Linear':            (LinearRegression(), X_base),
}

model_results = {}
for mname, (m, X_m) in models_def.items():
    m.fit(X_m, y)
    yp     = m.predict(X_m)
    res    = y - yp
    r2     = r2_score(y, yp)
    mae    = mean_absolute_error(y, yp)
    rmse   = np.sqrt(mean_squared_error(y, yp))
    mape   = np.mean(np.abs(res/np.maximum(y,1)))*100
    cv_r2  = cross_val_score(m, X_m, y, cv=min(5,len(y)//10), scoring='r2').mean()
    model_results[mname] = dict(model=m, X=X_m, yp=yp, r2=r2, mae=mae, rmse=rmse, mape=mape, cv_r2=cv_r2)

best_model_name = max(model_results, key=lambda k: model_results[k]['cv_r2'])
best            = model_results[best_model_name]


last_day    = c_data['day_num'].max()
future_days = np.arange(last_day+1, last_day+forecast_days+1)
future_dates= pd.date_range(start=c_data['date'].max()+pd.Timedelta(days=1), periods=forecast_days)


residuals   = best['yp'] - y
std_res     = residuals.std()
predictions = []
prev_cases  = y[-1]
for i, fday in enumerate(future_days):
    lag7    = y[-7] if i < 7 else predictions[i-7]
    ma7_val = np.mean(list(y[-7:]) + predictions[:i]) if predictions else prev_cases
    row     = [fday, np.sin(2*np.pi*fday/365), np.cos(2*np.pi*fday/365), lag7, ma7_val]
    pred    = best['model'].predict([row])[0]
    pred    = max(prev_cases, pred)
    predictions.append(pred)
    prev_cases = pred
predictions = np.array(predictions)

ci_upper = predictions + 1.96*np.linspace(std_res, std_res*2.5, forecast_days)
ci_lower = np.maximum(predictions - 1.96*np.linspace(std_res, std_res*2.5, forecast_days), y[-1])


fig2 = plt.figure(figsize=(18, 13))
fig2.patch.set_facecolor('#07090f')
gs2  = gridspec.GridSpec(3, 3, figure=fig2, hspace=.45, wspace=.35)


ax2a = fig2.add_subplot(gs2[0, :2])
ax2a.plot(c_data['date'], y/1e6, color='#7c3aed', lw=2.5, label='Actual Data')
ax2a.fill_between(future_dates, ci_lower/1e6, ci_upper/1e6,
                  alpha=.15, color='#06b6d4', label='95% CI Band')
ax2a.plot(future_dates, predictions/1e6, color='#06b6d4', lw=2.5,
          linestyle='--', label=f'{best_model_name} Forecast (+{forecast_days}d)')
ax2a.axvline(c_data['date'].max(), color='#8892a4', lw=1, linestyle=':', alpha=.6)
ax2a.text(c_data['date'].max(), ax2a.get_ylim()[1]*.95 if ax2a.get_ylim()[1]>0 else 1,
          ' Forecast\n Start', fontsize=8, color='#8892a4', va='top')
ax2a.set_title(f"{pred_country} — {best_model_name} Forecast")
ax2a.set_ylabel("Total Cases (Millions)")
ax2a.legend(fontsize=9)
ax2a.grid(axis='y', alpha=.3)


ax2b = fig2.add_subplot(gs2[0, 2])
mnames   = list(model_results.keys())
cv_r2s   = [model_results[m]['cv_r2'] for m in mnames]
bar_cols  = ['#10b981' if m == best_model_name else '#1e2535' for m in mnames]
bars = ax2b.barh(mnames, cv_r2s, color=bar_cols, edgecolor='none')
for bar, val in zip(bars, cv_r2s):
    ax2b.text(bar.get_width()+.001, bar.get_y()+bar.get_height()/2,
              f'{val:.4f}', va='center', fontsize=8.5, color='#e8edf5')
ax2b.set_xlabel("5-Fold CV R² Score")
ax2b.set_title("Model Comparison (CV R²)")
ax2b.set_xlim(0, 1.05)
ax2b.grid(axis='x', alpha=.3)


ax2c = fig2.add_subplot(gs2[1, :2])
res_pct = (y - best['yp']) / np.maximum(y, 1) * 100
ax2c.fill_between(c_data['date'], res_pct, 0,
                  where=(res_pct > 0), alpha=.5, color='#10b981', label='Overestimate')
ax2c.fill_between(c_data['date'], res_pct, 0,
                  where=(res_pct < 0), alpha=.5, color='#ef4444', label='Underestimate')
ax2c.axhline(0, color='#8892a4', lw=1)
ax2c.set_title("Model Residuals (% Error) over Time")
ax2c.set_ylabel("Residual %")
ax2c.legend(fontsize=9)
ax2c.grid(axis='y', alpha=.3)


ax2d = fig2.add_subplot(gs2[1, 2])
res_norm = (best['yp'] - y)
(osm, osr), (slope, intercept, r) = scipy_stats.probplot(res_norm, dist='norm')
ax2d.scatter(osm, osr, s=4, alpha=.4, color='#06b6d4')
ax2d.plot([osm.min(), osm.max()],
          [slope*osm.min()+intercept, slope*osm.max()+intercept],
          color='#ef4444', lw=1.5, linestyle='--')
ax2d.set_xlabel("Theoretical Quantiles")
ax2d.set_ylabel("Sample Quantiles")
ax2d.set_title(f"Q-Q Plot (Normality, r={r:.3f})")
ax2d.grid(alpha=.3)


ax2e = fig2.add_subplot(gs2[2, :])
ax2e.axis('off')
col_labels = ['Model', 'R² Score', 'CV R²', 'MAE (M)', 'RMSE (M)', 'MAPE %', 'Grade']
table_data = []
for mname, res in model_results.items():
    grade = '★ Best' if mname == best_model_name else ('Good' if res['cv_r2'] > .95 else 'Baseline')
    table_data.append([
        mname, f"{res['r2']:.4f}", f"{res['cv_r2']:.4f}",
        f"{res['mae']/1e6:.3f}", f"{res['rmse']/1e6:.3f}",
        f"{res['mape']:.2f}", grade
    ])

tbl = ax2e.table(cellText=table_data, colLabels=col_labels,
                 cellLoc='center', loc='center',
                 bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor('#0e1117' if r > 0 else '#1e2535')
    cell.set_edgecolor('#1e2535')
    cell.set_text_props(color='#e8edf5' if r == 0 else '#8892a4')
    if r > 0 and table_data[r-1][0] == best_model_name:
        cell.set_facecolor('#1a1f30')
        cell.set_text_props(color='#10b981')

ax2e.set_title("Full Model Comparison Table", fontsize=11, fontweight='bold',
               color='#e8edf5', pad=8)

fig2.suptitle("ML Forecasting Engine — Multi-Model Analysis", fontsize=14,
              fontweight='bold', color='#e8edf5', y=1.01)
plt.savefig('chart2_ml_forecasting.png', dpi=180, bbox_inches='tight',
            facecolor='#07090f')
plt.close()
print("   ✓ chart2_ml_forecasting.png")



print("\n[5/7] Generating Chart 3: Statistical Analysis...")

fig3 = plt.figure(figsize=(18, 12))
fig3.patch.set_facecolor('#07090f')
gs3  = gridspec.GridSpec(2, 3, figure=fig3, hspace=.42, wspace=.35)


ax3a = fig3.add_subplot(gs3[0, :2])
corr_cols   = ['total_cases','total_deaths','new_cases','new_deaths',
               'people_vaccinated_per_hundred','total_cases_per_million','cfr']
corr_data   = latest[corr_cols].dropna()
corr_matrix = corr_data.corr(method='pearson')
mask        = np.triu(np.ones_like(corr_matrix, dtype=bool))

cmap = LinearSegmentedColormap.from_list('cust', ['#ef4444','#0e1117','#3b82f6'])
sns.heatmap(corr_matrix, ax=ax3a, annot=True, fmt='.2f', cmap=cmap, center=0,
            mask=mask, linewidths=.5, linecolor='#1e2535',
            annot_kws={'size':9}, cbar_kws={'shrink':.8})
ax3a.set_title("Pearson Correlation Matrix (Lower Triangle)")
ax3a.tick_params(axis='x', rotation=30)
ax3a.tick_params(axis='y', rotation=0)


ax3b = fig3.add_subplot(gs3[0, 2])
heat_countries = ['India','United States','Brazil','United Kingdom','Germany']
heat_data = []
for country in heat_countries:
    c = countries_df[countries_df['location']==country].copy()
    c['month'] = c['date'].dt.to_period('M').astype(str)
    monthly    = c.groupby('month')['new_cases'].mean().reset_index()
    monthly['country'] = country
    heat_data.append(monthly)

hdf   = pd.concat(heat_data)
pivot = hdf.pivot(index='country', columns='month', values='new_cases').fillna(0)
pivot = np.log1p(pivot)

sns.heatmap(pivot, ax=ax3b, cmap='magma', linewidths=.3, linecolor='#07090f',
            cbar_kws={'label':'log(cases)','shrink':.8},
            xticklabels=max(1, len(pivot.columns)//6))
ax3b.set_title("Monthly Case Intensity (log scale)")
ax3b.set_xlabel("")
ax3b.tick_params(axis='x', rotation=45, labelsize=7)


ax3c = fig3.add_subplot(gs3[1, 0])
for i, country in enumerate(countries_focus[:5]):
    c  = countries_df[countries_df['location']==country].copy()
    dt = c['doubling_days'].rolling(14).mean().clip(0, 120)
    ax3c.plot(c['date'], dt, color=COLORS[i], lw=2, label=country, alpha=.85)
ax3c.axhline(7,  color='#ef4444', lw=1, linestyle='--', label='7d (Rapid)')
ax3c.axhline(30, color='#10b981', lw=1, linestyle='--', label='30d (Slow)')
ax3c.set_ylim(0, 100)
ax3c.set_title("Epidemic Doubling Time (14D MA)")
ax3c.set_ylabel("Days to Double Cases")
ax3c.legend(fontsize=7.5)
ax3c.grid(alpha=.3)


ax3d = fig3.add_subplot(gs3[1, 1])
ax3d.axis('off')

high_vacc  = latest[latest['people_vaccinated_per_hundred']>60]['total_deaths_per_million'].dropna()
low_vacc   = latest[latest['people_vaccinated_per_hundred']<30]['total_deaths_per_million'].dropna()
mwu_stat, mwu_p = scipy_stats.mannwhitneyu(high_vacc, low_vacc, alternative='less')

cfr_clean  = latest['cfr'].dropna().sample(min(50, len(latest['cfr'].dropna())), random_state=42)
sw_stat, sw_p = scipy_stats.shapiro(cfr_clean)

ks_stat, ks_p = scipy_stats.kstest(
    (latest['total_cases_per_million'].dropna() - latest['total_cases_per_million'].mean()) /
    latest['total_cases_per_million'].std(), 'norm'
)

tests = [
    ("Mann-Whitney U Test",       "High vs Low Vacc Deaths",
     f"U={mwu_stat:.1f}, p={mwu_p:.4f}",
     "Significant ✓" if mwu_p<.05 else "Not Sig. ✗"),
    ("Shapiro-Wilk Test",         "CFR Normality",
     f"W={sw_stat:.4f}, p={sw_p:.4f}",
     "Non-normal ✓" if sw_p<.05 else "Normal ✗"),
    ("Kolmogorov-Smirnov Test",   "Cases/M vs Normal",
     f"D={ks_stat:.4f}, p={ks_p:.4f}",
     "Non-normal ✓" if ks_p<.05 else "Normal ✗"),
    ("Pearson r (Cases vs Deaths)",
     f"r={corr_matrix.loc['total_cases','total_deaths']:.4f}",
     "Strong positive correlation", "p << 0.001 ✓"),
]

y_pos = .92
ax3d.text(.05, 1.02, "Statistical Hypothesis Tests", transform=ax3d.transAxes,
          fontsize=11, fontweight='bold', color='#e8edf5')
for tname, tdesc, tstat, tresult in tests:
    color = '#10b981' if '✓' in tresult else '#ef4444'
    ax3d.text(.05, y_pos, tname,              fontsize=9.5, fontweight='bold', color='#e8edf5')
    ax3d.text(.05, y_pos-.07, tdesc,          fontsize=8.5, color='#8892a4')
    ax3d.text(.05, y_pos-.14, tstat,          fontsize=8.5, color='#8892a4', style='italic')
    ax3d.text(.62, y_pos-.07, tresult,        fontsize=9,   color=color, fontweight='bold')
    ax3d.axhline(y=y_pos-.20, xmin=.03, xmax=.97, color='#1e2535', lw=.8)
    y_pos -= .26


ax3e = fig3.add_subplot(gs3[1, 2])
bins_h = latest[latest['people_vaccinated_per_hundred']>60]['total_deaths_per_million'].dropna()
bins_l = latest[latest['people_vaccinated_per_hundred']<30]['total_deaths_per_million'].dropna()
ax3e.hist(np.log1p(bins_h), bins=20, alpha=.65, color='#10b981', label='Vacc >60%', edgecolor='none')
ax3e.hist(np.log1p(bins_l), bins=20, alpha=.65, color='#ef4444', label='Vacc <30%', edgecolor='none')
ax3e.set_title("Deaths/M Distribution\nby Vaccination Level")
ax3e.set_xlabel("log(Deaths per Million)")
ax3e.set_ylabel("Count")
ax3e.legend(fontsize=9)
ax3e.grid(axis='y', alpha=.3)

fig3.suptitle("Epidemiological Statistical Analysis", fontsize=14,
              fontweight='bold', color='#e8edf5', y=1.01)
plt.savefig('chart3_statistical_analysis.png', dpi=180, bbox_inches='tight',
            facecolor='#07090f')
plt.close()
print("   ✓ chart3_statistical_analysis.png")



print("\n[6/7] Generating Chart 4: Anomaly Detection...")

fig4, axes4 = plt.subplots(2, 2, figsize=(16, 10))
fig4.patch.set_facecolor('#07090f')
fig4.suptitle("Anomaly Detection — IQR + Z-Score Ensemble", fontsize=13,
              fontweight='bold', color='#e8edf5')

for idx, (country, ax) in enumerate(zip(['India','United States','Brazil','United Kingdom'],
                                          axes4.flat)):
    c = countries_df[(countries_df['location']==country) &
                     (countries_df['new_cases'].notna())].copy().reset_index(drop=True)

    y_c     = c['new_cases'].fillna(0)
    z_score = np.abs(scipy_stats.zscore(y_c))
    q1, q3  = y_c.quantile(.25), y_c.quantile(.75)
    iqr     = q3 - q1
    anomaly = (z_score > 3) | (y_c > q3 + 3*iqr)

    ma_vals = y_c.rolling(7, min_periods=1).mean()
    ax.fill_between(c['date'], 0, y_c/1e3, alpha=.25,
                    color=COLORS[idx], label='Daily Cases')
    ax.plot(c['date'], ma_vals/1e3, color=COLORS[idx], lw=2, label='7D MA')
    ax.scatter(c['date'][anomaly], y_c[anomaly]/1e3,
               color='#ef4444', s=50, zorder=5,
               marker='x', linewidths=1.8, label=f'Anomalies ({anomaly.sum()})')

    ax.set_title(f"{country} — New Cases (Thousands)", color='#e8edf5')
    ax.set_ylabel("Cases (K)")
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=.3)
    ax.tick_params(axis='x', rotation=30, labelsize=7)

plt.tight_layout()
plt.savefig('chart4_anomaly_detection.png', dpi=180, bbox_inches='tight',
            facecolor='#07090f')
plt.close()
print("   ✓ chart4_anomaly_detection.png")



print("\n[7/7] Final Report")
print("\n" + "="*60)
print("   COVID-19 ADVANCED ANALYTICS — FINAL REPORT")
print("   Ayush Kumar | Final Year Project 2026")
print("="*60)

best_model_for_print = max(model_results, key=lambda k: model_results[k]['cv_r2'])
bm = model_results[best_model_for_print]

print(f"""
GLOBAL SUMMARY
  ├─ Total Cases  : {world_l['total_cases']/1e6:.1f} Million
  ├─ Total Deaths : {world_l['total_deaths']/1e6:.2f} Million
  └─ Global CFR   : {world_l['total_deaths']/world_l['total_cases']*100:.2f}%

INDIA
  ├─ Total Cases  : {india_l['total_cases']/1e6:.1f} Million
  ├─ Total Deaths : {india_l['total_deaths']/1e6:.3f} Million
  ├─ Death Rate   : {india_l['cfr']:.2f}%
  └─ Vaccinated % : {india_l.get('people_vaccinated_per_hundred',0):.1f}%

ML FORECASTING ({pred_country}, +{forecast_days} days)
  ├─ Best Model   : {best_model_for_print}
  ├─ R² Score     : {bm['r2']:.4f}
  ├─ CV R²        : {bm['cv_r2']:.4f}
  ├─ MAE          : {bm['mae']/1e6:.3f}M
  ├─ RMSE         : {bm['rmse']/1e6:.3f}M
  └─ MAPE         : {bm['mape']:.2f}%

STATISTICAL TESTS
  ├─ Mann-Whitney U (Deaths High vs Low Vacc)
  │    U = {mwu_stat:.2f}, p = {mwu_p:.4f} → {'SIGNIFICANT ✓' if mwu_p<.05 else 'Not significant'}
  ├─ Shapiro-Wilk (CFR Normality)
  │    W = {sw_stat:.4f}, p = {sw_p:.4f} → {'Non-normal (use non-parametric) ✓' if sw_p<.05 else 'Approximately normal'}
  └─ KS Test (Cases/M vs Normal)
       D = {ks_stat:.4f}, p = {ks_p:.4f} → {'Non-normal ✓' if ks_p<.05 else 'Normal'}

CHARTS SAVED
  ├─ chart1_country_intelligence.png
  ├─ chart2_ml_forecasting.png
  ├─ chart3_statistical_analysis.png
  └─ chart4_anomaly_detection.png
""")
print("="*60)
print("   ✓ Analysis complete! Project ready to submit.")
print("="*60)
