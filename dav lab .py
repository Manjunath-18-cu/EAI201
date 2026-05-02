{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "d612d952",
   "metadata": {},
   "source": [
    "# NYC Taxi Ecosystem — Investigative Analytics Study\n",
    "Full Python Notebook with Visualizations & Interpretations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "86460cc1",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy import stats\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "519bad4f","""
NYC TAXI ECOSYSTEM — FULL INVESTIGATIVE ANALYTICS STUDY
Roll: 0,3,1,9 | r=4 | Window: Jan–Mar 2023
Author: Senior Data Analyst
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ACT 1: MATHEMATICAL FRAMING
# ============================================================
digits = [0, 3, 1, 9]
digit_sum = sum(digits)

def digital_root(n):
    while n >= 10:
        n = sum(int(d) for d in str(n))
    return n

r = digital_root(digit_sum)
# f(x) = x^3/3 - rx^2 + (r^2-1)x  =>  f'(x) = x^2 - 2rx + (r^2-1) = 0
# discriminant = 4r^2 - 4(r^2-1) = 4  =>  x = r ± 1
x1, x2 = r - 1, r + 1  # = 3, 5  =>  months Jan (3) to Mar (5)
print(f"r={r}, critical points: x1={x1} (Jan, local max), x2={x2} (Mar, local min)")

# ============================================================
# ACT 2: DATA CONSTRUCTION
# ============================================================
np.random.seed(319)

def generate_nyc_taxi_data(n=300000):
    rng = np.random.default_rng(319)
    from datetime import datetime, timedelta
    ts_start = datetime(2023, 1, 1)
    ts_end   = datetime(2023, 3, 28)
    delta_s  = int((ts_end - ts_start).total_seconds())
    raw_s    = rng.integers(0, delta_s, n)
    pickup_ts = pd.Series([ts_start + timedelta(seconds=int(s)) for s in raw_s])

    distances    = rng.lognormal(0.8, 0.7, n)
    speed_mph    = rng.normal(12, 4, n).clip(2, 40)
    duration_min = (distances / speed_mph * 60 + rng.exponential(2, n)).clip(1, 180)
    base_fare    = 3.0 + 2.50 * distances + 0.50 * duration_min / 60 * 60
    tips         = base_fare * rng.beta(2, 5, n) * 0.35
    total_amount = base_fare + tips + rng.choice([0.5, 1.0, 1.5], n)
    payment_type = rng.choice([1, 2, 3, 4], n, p=[0.65, 0.30, 0.03, 0.02])
    passenger_count = rng.choice([1,2,3,4,5,6], n, p=[0.58,0.20,0.10,0.06,0.04,0.02])
    dropoff_ts   = pickup_ts + pd.to_timedelta(duration_min, unit='m')

    # Inject 12% data quality issues
    n_issues = int(n * 0.12)
    issue_idx = rng.choice(n, n_issues, replace=False)
    distances_a = distances.copy().astype(float)
    fare_a = base_fare.copy(); tip_a = tips.copy(); total_a = total_amount.copy()
    pc_a = passenger_count.copy().astype(float)
    pickup_a = pickup_ts.values.copy(); dropoff_a = dropoff_ts.values.copy()
    pt_a = payment_type.copy(); dur_a = duration_min.copy()

    i1 = issue_idx[:int(n_issues*0.15)]; distances_a[i1] = 0.0
    i2 = issue_idx[int(n_issues*0.15):int(n_issues*0.25)]; fare_a[i2] = -abs(fare_a[i2]); total_a[i2] = fare_a[i2]
    i3 = issue_idx[int(n_issues*0.25):int(n_issues*0.33)]; dropoff_a[i3] = pickup_a[i3] - np.timedelta64(300,'s')
    i4 = issue_idx[int(n_issues*0.33):int(n_issues*0.43)]; pc_a[i4] = 0
    i5 = issue_idx[int(n_issues*0.43):int(n_issues*0.48)]; total_a[i5] = rng.uniform(500,5000,len(i5))
    i6 = issue_idx[int(n_issues*0.48):int(n_issues*0.60)]; distances_a[i6[:len(i6)//2]] = np.nan; pc_a[i6[len(i6)//2:]] = np.nan
    i7 = issue_idx[int(n_issues*0.60):int(n_issues*0.65)]; dur_a[i7] = rng.uniform(1441,10000,len(i7))
    i8 = issue_idx[int(n_issues*0.65):int(n_issues*0.70)]; pt_a[i8] = 2; tip_a[i8] = rng.uniform(1,20,len(i8))

    df = pd.DataFrame({
        'tpep_pickup_datetime': pd.to_datetime(pickup_a),
        'tpep_dropoff_datetime': pd.to_datetime(dropoff_a),
        'passenger_count': pc_a, 'trip_distance': distances_a,
        'fare_amount': fare_a, 'tip_amount': tip_a,
        'total_amount': total_a, 'payment_type': pt_a.astype(float), 'duration_min': dur_a,
    })
    return df

df_raw = generate_nyc_taxi_data()

# --- Valid trip criteria ---
mask_dist   = df_raw['trip_distance'].notna() & (df_raw['trip_distance'] > 0.1) & (df_raw['trip_distance'] <= 100)
mask_dur    = (df_raw['duration_min'] >= 1) & (df_raw['duration_min'] <= 180)
mask_fare   = (df_raw['fare_amount'] >= 2.50) & (df_raw['fare_amount'] <= 500)
mask_total  = df_raw['total_amount'] > 0
mask_time   = df_raw['tpep_dropoff_datetime'] > df_raw['tpep_pickup_datetime']
mask_pax    = df_raw['passenger_count'].notna() & df_raw['passenger_count'].between(1,6)
df_raw['calc_speed'] = df_raw['trip_distance'] / (df_raw['duration_min'] / 60)
mask_speed  = df_raw['calc_speed'].between(0, 80, inclusive='neither')
mask_cash   = ~((df_raw['payment_type'] == 2) & (df_raw['tip_amount'] > 0))
valid_mask  = mask_dist & mask_dur & mask_fare & mask_total & mask_time & mask_pax & mask_speed & mask_cash

df_clean = df_raw[valid_mask].copy()
df_clean['pickup_hour']  = df_clean['tpep_pickup_datetime'].dt.hour
df_clean['pickup_dow']   = df_clean['tpep_pickup_datetime'].dt.dayofweek
df_clean['pickup_month'] = df_clean['tpep_pickup_datetime'].dt.month
df_clean['pickup_date']  = df_clean['tpep_pickup_datetime'].dt.date
df_clean['revenue_per_min']  = df_clean['total_amount'] / df_clean['duration_min']
df_clean['revenue_per_mile'] = df_clean['total_amount'] / df_clean['trip_distance']
print(f"Clean: {len(df_clean):,} / {len(df_raw):,} ({len(df_clean)/len(df_raw)*100:.1f}%)")

# ============================================================
# ACT 3: HYPOTHESIS TESTING
# ============================================================
monthly = df_clean.groupby('pickup_month').agg(
    trips=('total_amount','count'), avg_fare=('fare_amount','mean'),
    avg_dist=('trip_distance','mean'), avg_dur=('duration_min','mean'),
    rpm=('revenue_per_min','mean'), total_rev=('total_amount','sum'),
).reset_index()
print("\nMonthly stats:\n", monthly.round(3).to_string(index=False))

jan_rpm = df_clean[df_clean['pickup_month']==1]['revenue_per_min']
mar_rpm = df_clean[df_clean['pickup_month']==3]['revenue_per_min']
jan_c = jan_rpm[jan_rpm < jan_rpm.quantile(0.99)]
mar_c = mar_rpm[mar_rpm < mar_rpm.quantile(0.99)]
t, p = stats.ttest_ind(jan_c, mar_c)
d = (jan_c.mean()-mar_c.mean()) / np.sqrt((jan_c.std()**2+mar_c.std()**2)/2)
print(f"\nT-test: t={t:.4f}, p={p:.6f}, Cohen's d={d:.4f}")
print("Conclusion:", "REJECT H0" if p < 0.05 else "FAIL TO REJECT H0 — RPM is stable")

# ============================================================
# ACT 5: SIMULATION
# ============================================================
elasticities = {'optimistic': -0.20, 'moderate': -0.40, 'pessimistic': -0.65}
fare_increase = r / 100
baseline_rev  = df_clean['total_amount'].sum()
baseline_n    = len(df_clean)
baseline_fare = df_clean['fare_amount'].mean()
print("\n4% Fare Increase Simulation:")
for scenario, e in elasticities.items():
    new_n    = baseline_n * (1 + e * fare_increase)
    new_fare = baseline_fare * (1 + fare_increase)
    new_rev  = new_n * (baseline_rev/baseline_n * (1 + fare_increase))
    delta    = (new_rev - baseline_rev)/baseline_rev*100
    print(f"  {scenario:<15}: revenue {delta:+.2f}%")

# ============================================================
# SQL EQUIVALENTS (printed for reference)
# ============================================================
SQL = """
-- ============================================================
-- SQL: Data Cleaning (NYC TLC Yellow Taxi)
-- ============================================================
CREATE TABLE nyc_taxi_clean AS
SELECT *,
    (EXTRACT(EPOCH FROM tpep_dropoff_datetime - tpep_pickup_datetime)/60) AS duration_min,
    trip_distance / NULLIF(EXTRACT(EPOCH FROM tpep_dropoff_datetime - tpep_pickup_datetime)/3600, 0) AS avg_speed_mph,
    total_amount / NULLIF(EXTRACT(EPOCH FROM tpep_dropoff_datetime - tpep_pickup_datetime)/60, 0) AS revenue_per_min,
    total_amount / NULLIF(trip_distance, 0) AS revenue_per_mile
FROM nyc_taxi_raw
WHERE
    trip_distance > 0.1 AND trip_distance <= 100
    AND fare_amount >= 2.50 AND fare_amount <= 500
    AND total_amount > 0
    AND tpep_dropoff_datetime > tpep_pickup_datetime
    AND EXTRACT(EPOCH FROM tpep_dropoff_datetime - tpep_pickup_datetime)/60 BETWEEN 1 AND 180
    AND passenger_count BETWEEN 1 AND 6
    AND passenger_count IS NOT NULL
    AND trip_distance IS NOT NULL
    AND NOT (payment_type = 2 AND tip_amount > 0);

-- ============================================================
-- SQL: Monthly Aggregation
-- ============================================================
SELECT
    EXTRACT(MONTH FROM tpep_pickup_datetime) AS pickup_month,
    COUNT(*) AS total_trips,
    ROUND(AVG(fare_amount)::numeric, 2) AS avg_fare,
    ROUND(AVG(trip_distance)::numeric, 3) AS avg_distance_miles,
    ROUND(AVG(duration_min)::numeric, 2) AS avg_duration_min,
    ROUND(AVG(revenue_per_min)::numeric, 4) AS avg_rpm,
    ROUND(SUM(total_amount)::numeric, 2) AS total_revenue
FROM nyc_taxi_clean
GROUP BY pickup_month
ORDER BY pickup_month;

-- ============================================================
-- SQL: Hourly Pattern
-- ============================================================
SELECT
    EXTRACT(HOUR FROM tpep_pickup_datetime) AS hour_of_day,
    COUNT(*) AS trip_count,
    ROUND(AVG(fare_amount)::numeric, 2) AS avg_fare,
    ROUND(AVG(revenue_per_min)::numeric, 4) AS avg_rpm
FROM nyc_taxi_clean
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- ============================================================
-- SQL: Removal Audit
-- ============================================================
SELECT
    SUM(CASE WHEN trip_distance IS NULL OR trip_distance <= 0.1 OR trip_distance > 100 THEN 1 ELSE 0 END) AS invalid_distance,
    SUM(CASE WHEN fare_amount < 2.50 OR fare_amount > 500 THEN 1 ELSE 0 END) AS invalid_fare,
    SUM(CASE WHEN total_amount <= 0 THEN 1 ELSE 0 END) AS negative_total,
    SUM(CASE WHEN tpep_dropoff_datetime <= tpep_pickup_datetime THEN 1 ELSE 0 END) AS dropoff_before_pickup,
    SUM(CASE WHEN passenger_count IS NULL OR passenger_count < 1 OR passenger_count > 6 THEN 1 ELSE 0 END) AS invalid_passengers,
    SUM(CASE WHEN payment_type = 2 AND tip_amount > 0 THEN 1 ELSE 0 END) AS cash_tip_contradiction,
    COUNT(*) AS total_raw
FROM nyc_taxi_raw;
"""
print("\nSQL queries available in output file.")


   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "np.random.seed(42)\n",
    "\n",
    "dates = pd.date_range(start=\"2024-01-01\", end=\"2024-03-31\")\n",
    "data = []\n",
    "\n",
    "for d in dates:\n",
    "    trips = np.random.randint(2000, 3200)\n",
    "    revenue = trips * np.random.uniform(15, 22)\n",
    "    avg_fare = revenue / trips\n",
    "\n",
    "    data.append([d, trips, revenue, avg_fare])\n",
    "\n",
    "df = pd.DataFrame(data, columns=[\"Date\",\"Trips\",\"Revenue\",\"AvgFare\"])\n",
    "df[\"Month\"] = df[\"Date\"].dt.month\n",
    "df[\"Hour\"] = np.random.randint(0,24,len(df))\n",
    "\n",
    "df.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a2630b51",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "monthly = df.groupby(\"Month\").agg({\n",
    "    \"Trips\":\"sum\",\n",
    "    \"Revenue\":\"sum\",\n",
    "    \"AvgFare\":\"mean\"\n",
    "}).reset_index()\n",
    "\n",
    "monthly\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4c1aff67",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "plt.figure()\n",
    "plt.plot(monthly[\"Month\"], monthly[\"Trips\"], marker='o')\n",
    "plt.title(\"Monthly Trip Volume\")\n",
    "plt.xlabel(\"Month\")\n",
    "plt.ylabel(\"Trips\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0328446a",
   "metadata": {},
   "source": [
    "### Interpretation\n",
    "Trip volume shows a declining or fluctuating trend indicating possible demand instability."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a1b84f3b",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "plt.figure()\n",
    "plt.plot(monthly[\"Month\"], monthly[\"AvgFare\"], marker='o')\n",
    "plt.title(\"Average Fare (Stable)\")\n",
    "plt.xlabel(\"Month\")\n",
    "plt.ylabel(\"Fare\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e33f0831",
   "metadata": {},
   "source": [
    "### Interpretation\n",
    "Average fare remains stable, which can create a misleading perception of system health."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6ba275c0",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "df[\"Rolling\"] = df[\"Trips\"].rolling(7).mean()\n",
    "\n",
    "plt.figure()\n",
    "plt.plot(df[\"Date\"], df[\"Trips\"], alpha=0.4, label=\"Raw\")\n",
    "plt.plot(df[\"Date\"], df[\"Rolling\"], label=\"7-day Avg\")\n",
    "plt.legend()\n",
    "plt.title(\"Signal vs Noise\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6d5c837b",
   "metadata": {},
   "source": [
    "### Interpretation\n",
    "Rolling average reveals the true trend, filtering daily fluctuations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "69372293",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "jan = df[df[\"Month\"]==1][\"Trips\"]\n",
    "mar = df[df[\"Month\"]==3][\"Trips\"]\n",
    "\n",
    "t,p = stats.ttest_ind(jan, mar)\n",
    "\n",
    "print(\"t-stat:\", t)\n",
    "print(\"p-value:\", p)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c56d1e93",
   "metadata": {},
   "source": [
    "### Interpretation\n",
    "If p < 0.05, the difference is statistically significant."
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
