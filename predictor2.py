# =============================================================================
#  AIR QUALITY FORECASTING SCRIPT
#  Arima: CO + O3 forecast
#  Point Lisas: O3 only (no CO sensor data available)
# =============================================================================

# -----------------IMPORTATIONS------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1 — LOAD AND CLEAN DATA
# =============================================================================

station_a = pd.read_csv('arima_station_data.csv')
station_b = pd.read_csv('point_lisas_data.csv')

# Drop the empty extra column in the Arima file if it exists
station_a = station_a.drop(columns=[c for c in station_a.columns if 'Unnamed' in c])

# Fix 'Sept' typo → 'Sep' so pandas can parse it
station_a['Date'] = station_a['Date'].str.replace('Sept', 'Sep', regex=False)
station_b['Date'] = station_b['Date'].str.replace('Sept', 'Sep', regex=False)

# Parse dates
station_a['Date'] = pd.to_datetime(station_a['Date'], format='%d-%b-%y')
station_b['Date'] = pd.to_datetime(station_b['Date'], format='%d-%b-%y')

# Convert 'n/a' strings to proper NaN, then force numeric
station_a['CO'] = pd.to_numeric(station_a['CO'], errors='coerce')
station_a['O3'] = pd.to_numeric(station_a['O3'], errors='coerce')
station_b['CO'] = pd.to_numeric(station_b['CO'], errors='coerce')  # will be all NaN
station_b['O3'] = pd.to_numeric(station_b['O3'], errors='coerce')

# Point Lisas has no CO sensor . fill with 0 as a placeholder
# AQI for Point Lisas will be based on O3 only
station_b['CO'] = 0.0

# Fill O3 gaps using forward-fill then backward-fill
station_a['O3'] = station_a['O3'].ffill().bfill()
station_b['O3'] = station_b['O3'].ffill().bfill()

# Track which stations have real CO data
co_available  = {'Arima': True, 'Point Lisas': False}
station_names = ['Arima', 'Point Lisas']

print("Arima shape:", station_a.shape, "| Missing:", station_a[['CO','O3']].isnull().sum().to_dict())
print("Point Lisas shape:", station_b.shape, "| Missing:", station_b[['CO','O3']].isnull().sum().to_dict())


# =============================================================================
# SECTION 2 — AQI HELPER FUNCTIONS
# =============================================================================

def co_to_aqi(co_mgm3):
    """Convert CO (mg/m³) to US EPA AQI. Returns (aqi, category, hex_colour)."""
    breakpoints = [
        (0,    4.4,   0,   50,  'Good',               '#00e400'),
        (4.5,  9.4,   51,  100, 'Moderate',           '#ffff00'),
        (9.5,  12.4,  101, 150, 'Unhealthy for Some', '#ff7e00'),
        (12.5, 15.4,  151, 200, 'Unhealthy',          '#ff0000'),
        (15.5, 30.4,  201, 300, 'Very Unhealthy',     '#8f3f97'),
        (30.5, 50.4,  301, 500, 'Hazardous',          '#7e0023'),
    ]
    v = co_mgm3 * 0.873  # mg/m³ → ppm
    for lo, hi, ilo, ihi, label, col in breakpoints:
        if lo <= v <= hi:
            return round(((ihi - ilo) / (hi - lo)) * (v - lo) + ilo), label, col
    return 500, 'Hazardous', '#7e0023'


def o3_to_aqi(o3_ugm3):
    """Convert O3 (µg/m³) to US EPA AQI. Returns (aqi, category, hex_colour)."""
    breakpoints = [
        (0,   54,  0,   50,  'Good',               '#00e400'),
        (55,  70,  51,  100, 'Moderate',           '#ffff00'),
        (71,  85,  101, 150, 'Unhealthy for Some', '#ff7e00'),
        (86,  105, 151, 200, 'Unhealthy',          '#ff0000'),
        (106, 200, 201, 300, 'Very Unhealthy',     '#8f3f97'),
    ]
    v = o3_ugm3 * 0.5  # µg/m³ → ppb
    for lo, hi, ilo, ihi, label, col in breakpoints:
        if lo <= v <= hi:
            return round(((ihi - ilo) / (hi - lo)) * (v - lo) + ilo), label, col
    return 300, 'Very Unhealthy', '#8f3f97'


def overall_aqi(co_val, o3_val, has_co=True):
    """
    Overall AQI = max of individual pollutant AQIs (US EPA method).
    If has_co=False (Point Lisas), uses O3 only.
    """
    o3_aqi, o3_cat, o3_col = o3_to_aqi(o3_val)
    if not has_co:
        return o3_aqi, 'O3', o3_cat, o3_col
    co_aqi, co_cat, co_col = co_to_aqi(co_val)
    if co_aqi >= o3_aqi:
        return co_aqi, 'CO', co_cat, co_col
    return o3_aqi, 'O3', o3_cat, o3_col


# =============================================================================
# SECTION 3 — FEATURE ENGINEERING
# =============================================================================

FEATURE_COLS = [
    'CO', 'O3',
    'CO_lag1', 'CO_lag2', 'CO_lag7',
    'O3_lag1', 'O3_lag2', 'O3_lag7',
    'CO_roll7', 'O3_roll7',
    'day_of_year', 'month', 'day_of_week'
]

def build_features(df):
    """
    Create lag and rolling features for CO and O3.
    Lag features give the model historical context for time series forecasting.
    """
    df = df.copy().sort_values('Date').reset_index(drop=True)

    # Lag features — past values at 1, 2, and 7 days back
    for col in ['CO', 'O3']:
        df[f'{col}_lag1']  = df[col].shift(1)
        df[f'{col}_lag2']  = df[col].shift(2)
        df[f'{col}_lag7']  = df[col].shift(7)
        df[f'{col}_roll7'] = df[col].shift(1).rolling(7).mean()  # 7-day trailing average

    # Calendar features to capture seasonality
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['month']       = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday

    # Targets: what we want to predict (next day's values)
    df['CO_tmrw'] = df['CO'].shift(-1)
    df['O3_tmrw'] = df['O3'].shift(-1)

    df = df.dropna()  # Remove rows where lags produced NaN
    return df


# =============================================================================
# SECTION 4 — MODEL TRAINING WITH TIME SERIES CROSS-VALIDATION
# =============================================================================

def train_model(df, target_col):
    """
    Train an XGBoost model for one pollutant target.
    Uses TimeSeriesSplit — each fold only trains on past data (no leakage).
    Returns: (trained_model, average_MAE_across_folds)
    """
    X = df[FEATURE_COLS]
    y = df[target_col]

    tscv = TimeSeriesSplit(n_splits=5)
    maes = []

    # Cross-validation loop — trains on past, validates on future
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m = XGBRegressor(
            n_estimators=100,    # number of trees
            learning_rate=0.05,  # how much each tree corrects the last
            max_depth=5,         # tree depth — limits overfitting
            subsample=0.8,       # random row sample per tree
            random_state=42
        )
        m.fit(X_tr, y_tr)
        maes.append(mean_absolute_error(y_val, m.predict(X_val)))

    print(f"  [{target_col}] CV MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")

    # Final model: retrain on ALL data for best forecast performance
    final_model = XGBRegressor(
        n_estimators=100, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=42
    )
    final_model.fit(X, y)
    return final_model, float(np.mean(maes))


# =============================================================================
# SECTION 5 — 30-DAY RECURSIVE FORECAST
# =============================================================================

def forecast_30_days(df, co_model, o3_model, has_co=True):
    """
    Predict CO and O3 for the next 30 days using a recursive strategy:
    predict Day 1 → append to history → predict Day 2 → repeat.
    Errors compound slightly over the horizon, so later days are less certain.
    """
    history   = df[['Date', 'CO', 'O3']].tail(30).copy().reset_index(drop=True)
    last_date = history['Date'].max()
    rows      = []

    for day in range(1, 31):
        next_date = last_date + pd.Timedelta(days=day)
        cs = history['CO']
        os = history['O3']

        # Build one feature row from the rolling history window
        row = {
            'CO':          cs.iloc[-1],
            'O3':          os.iloc[-1],
            'CO_lag1':     cs.iloc[-1],
            'CO_lag2':     cs.iloc[-2] if len(cs) >= 2 else cs.iloc[-1],
            'CO_lag7':     cs.iloc[-7] if len(cs) >= 7 else cs.iloc[0],
            'O3_lag1':     os.iloc[-1],
            'O3_lag2':     os.iloc[-2] if len(os) >= 2 else os.iloc[-1],
            'O3_lag7':     os.iloc[-7] if len(os) >= 7 else os.iloc[0],
            'CO_roll7':    cs.tail(7).mean(),
            'O3_roll7':    os.tail(7).mean(),
            'day_of_year': next_date.dayofyear,
            'month':       next_date.month,
            'day_of_week': next_date.dayofweek,
        }

        X_row   = pd.DataFrame([row])[FEATURE_COLS]
        co_pred = float(co_model.predict(X_row)[0])
        o3_pred = float(o3_model.predict(X_row)[0])

        aqi_val, dominant, category, colour = overall_aqi(co_pred, o3_pred, has_co)

        rows.append({
            'Date':     next_date,
            'CO_pred':  round(co_pred, 3) if has_co else None,
            'O3_pred':  round(o3_pred, 1),
            'AQI':      aqi_val,
            'Dominant': dominant,
            'Category': category,
            'Color':    colour,
        })

        # Append prediction back into history so the next day can use it
        history = pd.concat(
            [history, pd.DataFrame([{'Date': next_date, 'CO': co_pred, 'O3': o3_pred}])],
            ignore_index=True
        )

    return pd.DataFrame(rows)


# =============================================================================
# SECTION 6 — MAIN PIPELINE
# =============================================================================

results = {}

for station_df, station_name in zip([station_a, station_b], station_names):
    print(f"\n{'='*60}")
    print(f"Processing: {station_name}")
    print(f"{'='*60}")

    has_co = co_available[station_name]
    if not has_co:
        print("  NOTE: No CO sensor data — forecasting O3 only.")

    engineered = build_features(station_df)
    print(f"  Training rows: {len(engineered)}")

    co_model, co_mae = train_model(engineered, 'CO_tmrw')
    o3_model, o3_mae = train_model(engineered, 'O3_tmrw')

    forecast_df = forecast_30_days(engineered, co_model, o3_model, has_co)

    print(f"\n  30-Day Forecast Preview:")
    print(forecast_df[['Date', 'CO_pred', 'O3_pred', 'AQI', 'Category']].to_string(index=False))

    results[station_name] = {
        'engineered': engineered,
        'co_mae':     co_mae,
        'o3_mae':     o3_mae,
        'forecast':   forecast_df,
        'has_co':     has_co,
    }


# =============================================================================
# SECTION 7 — VISUALISATIONS
# =============================================================================

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('30-Day Air Quality Forecast — Arima vs Point Lisas',
             fontsize=16, fontweight='bold', y=0.98)

colours = {'Arima': '#d62728', 'Point Lisas': '#1f77b4'}

for col_idx, (name, res) in enumerate(results.items()):
    fc     = res['forecast']
    clr    = colours[name]
    has_co = res['has_co']

    # ── CO Forecast ───────────────────────────────────────────────────────────
    ax = axes[0][col_idx]
    if has_co:
        ax.plot(fc['Date'], fc['CO_pred'], color=clr, lw=2, marker='o', ms=3)
        ax.fill_between(fc['Date'], fc['CO_pred'] * 0.90, fc['CO_pred'] * 1.10,
                        alpha=0.2, color=clr)
        ax.set_title(f'{name}\nCO Forecast (mg/m³)', fontsize=11, fontweight='bold')
    else:
        ax.set_facecolor('#f5f5f5')
        ax.text(0.5, 0.5, 'CO data unavailable\n(no sensor at this station)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, color='#999999')
        ax.set_title(f'{name}\nCO — No Sensor Data', fontsize=11,
                     fontweight='bold', color='grey')
    ax.set_ylabel('CO (mg/m³)')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Date')

    # ── O3 Forecast ───────────────────────────────────────────────────────────
    ax = axes[1][col_idx]
    ax.plot(fc['Date'], fc['O3_pred'], color=clr, lw=2, marker='s', ms=3, ls='--')
    ax.fill_between(fc['Date'], fc['O3_pred'] * 0.90, fc['O3_pred'] * 1.10,
                    alpha=0.2, color=clr)
    ax.set_title(f'{name}\nO3 Forecast (µg/m³)', fontsize=11, fontweight='bold')
    ax.set_ylabel('O3 (µg/m³)')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Date')

    # ── AQI Colour Bar ────────────────────────────────────────────────────────
    ax = axes[2][col_idx]
    ax.bar(range(len(fc)), fc['AQI'], color=fc['Color'].tolist(),
           edgecolor='white', lw=0.5)
    title_suffix = ' (O3 only — no CO sensor)' if not has_co else ''
    ax.set_title(f'{name}\nDaily AQI{title_suffix}', fontsize=11, fontweight='bold')
    ax.set_ylabel('AQI Value')
    ax.set_xlabel('Day of Forecast')
    ax.set_xticks(range(0, 30, 5))
    ax.set_xticklabels([f'Day {i+1}' for i in range(0, 30, 5)])
    ax.grid(True, alpha=0.3, axis='y')
    legend_patches = [
        mpatches.Patch(color='#00e400', label='Good (0–50)'),
        mpatches.Patch(color='#ffff00', label='Moderate (51–100)'),
        mpatches.Patch(color='#ff7e00', label='Unhealthy for Some (101–150)'),
        mpatches.Patch(color='#ff0000', label='Unhealthy (151–200)'),
        mpatches.Patch(color='#8f3f97', label='Very Unhealthy (201–300)'),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc='upper right', framealpha=0.8)

plt.tight_layout()
plt.savefig('forecast_charts.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nCharts saved to forecast_charts.png")


# =============================================================================
# SECTION 8 — FORECAST TABLES
# =============================================================================

print("\n" + "="*70)
print("30-DAY FORECAST TABLE")
print("="*70)

for name, res in results.items():
    print(f"\n{name}")
    print("-"*70)
    fc = res['forecast'].copy()
    if not res['has_co']:
        fc['CO_pred'] = 'N/A'
    fc_display = fc[['Date', 'CO_pred', 'O3_pred', 'AQI', 'Category', 'Dominant']]
    fc_display.columns = ['Date', 'CO (mg/m³)', 'O3 (µg/m³)', 'AQI', 'AQI Category', 'Dominant']
    print(fc_display.to_string(index=False))
    aqi = res['forecast']['AQI']
    cat = res['forecast']['Category']
    print(f"\n  Avg AQI: {aqi.mean():.1f}  |  Max AQI: {aqi.max()}  |  "
          f"Days Good: {(cat=='Good').sum()}  |  Days Unhealthy+: {(aqi > 150).sum()}")

print("\nScript complete.")