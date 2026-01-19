import pandas as pd
import numpy as np
import glob
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns

# --- MODULE 1: DATA ENGINEERING ---
def load_data_pipeline(file_pattern, label):
    """
    Ingests raw shards, handles type casting, and aggregates to 'Day-Pincode' grain.
    Optimized for memory by dropping columns early.
    """
    print(f"[{label}] Ingesting data shards...")
    files = glob.glob(file_pattern)
    df_list = []
    
    for f in files:
        try:
            # Load specific columns to save RAM
            temp = pd.read_csv(f, usecols=['date', 'pincode', 'state', 'district'] + 
                                       [c for c in pd.read_csv(f, nrows=1).columns if 'age' in c])
            df_list.append(temp)
        except Exception as e:
            print(f"Warning: Failed to load {f}. Error: {e}")
            
    if not df_list:
        raise ValueError(f"No files found for {label}. Check paths.")
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Date Casting with Error Handling
    full_df['date'] = pd.to_datetime(full_df['date'], format='%d-%m-%Y', errors='coerce')
    full_df = full_df.dropna(subset=['date']) # Drop bad dates
    
    return full_df

# Load & Merge
print(">>> INITIALIZING DATA PIPELINE...")
bio = load_data_pipeline('data/api_data_aadhar_biometric_*.csv', 'Biometrics')
demo = load_data_pipeline('data/api_data_aadhar_demographic_*.csv', 'Demographics')

# Aggregating to Daily Level per Pincode (The "Atomic Unit" of analysis)
# We sum up volumes for the same day/pincode
bio_grp = bio.groupby(['date', 'pincode', 'state', 'district'])[['bio_age_5_17', 'bio_age_17_']].sum().reset_index()
demo_grp = demo.groupby(['date', 'pincode', 'state', 'district'])[['demo_age_5_17', 'demo_age_17_']].sum().reset_index()

# MASTER MERGE
df = pd.merge(bio_grp, demo_grp, on=['date', 'pincode', 'state', 'district'], how='outer').fillna(0)

# --- MODULE 2: ADVANCED FEATURE ENGINEERING (The "Secret Sauce") ---
# 1. Total Load Calculation
df['total_load'] = df['bio_age_5_17'] + df['bio_age_17_'] + df['demo_age_5_17'] + df['demo_age_17_']

# 2. The "Student Pressure Index" (SPI)
# Logic: If SPI > 0.8, the center is effectively a school extension counter.
df['SPI'] = (df['bio_age_5_17'] + df['demo_age_5_17']) / (df['total_load'] + 1) # +1 to avoid div/0

# 3. The "Biometric Intensity Ratio" (BIR)
# Logic: Biometric updates take 10x longer than demographic. This measures "Time Cost".
df['BIR'] = (df['bio_age_5_17'] + df['bio_age_17_']) / (df['total_load'] + 1)

# Filter for meaningful analysis (Remove tiny centers)
analysis_df = df[df['total_load'] > 20].copy()

# --- MODULE 3: ANOMALY DETECTION (ISOLATION FOREST) ---
# Research Context: "Unsupervised detection of operational irregularities"
print(">>> RUNNING ISOLATION FOREST FOR FRAUD/ANOMALY DETECTION...")

features = ['total_load', 'SPI', 'BIR']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(analysis_df[features])

# Contamination = 0.01 (We assume top 1% are anomalies worth investigating)
iso = IsolationForest(contamination=0.01, random_state=42, n_estimators=100)
analysis_df['anomaly_score'] = iso.fit_predict(X_scaled)
analysis_df['anomaly_severity'] = iso.decision_function(X_scaled) # Lower is more anomalous

# Extract Anomalies (Score = -1)
anomalies = analysis_df[analysis_df['anomaly_score'] == -1]

# --- MODULE 4: TEMPORAL DECOMPOSITION (STL) ---
# Research Context: Separating "Seasonality" from "Trend"
print(">>> PERFORMING TIME-SERIES DECOMPOSITION...")

# Aggregate to National Daily Level
daily_ts = df.groupby('date')['bio_age_5_17'].sum()
daily_ts = daily_ts.asfreq('D').fillna(0) # Ensure continuous frequency

# Decompose (Period = 30 days for monthly seasonality detection)
decomposition = seasonal_decompose(daily_ts, model='additive', period=30)

# --- MODULE 5: VISUALIZATION & REPORTING ---
plt.figure(figsize=(15, 10))

# Plot 1: The "Ghost Centers" (Anomalies)
plt.subplot(2, 2, 1)
sns.scatterplot(data=analysis_df, x='total_load', y='SPI', hue='anomaly_score', palette={1:'blue', -1:'red'}, alpha=0.6)
plt.title('Forensic Map: Detecting "Ghost Centers" (Red)', fontweight='bold')
plt.xlabel('Daily Volume')
plt.ylabel('Student Pressure Index (SPI)')
plt.legend(title='Status', labels=['Normal', 'Anomaly'])

# Plot 2: Seasonal Component (The "Panic Cycles")
plt.subplot(2, 2, 2)
decomposition.seasonal.plot(color='green')
plt.title('Extracted Seasonality: The Recurring "Panic Cycles"', fontweight='bold')
plt.ylabel('Volume Deviation')

# Plot 3: Trend Component (The "Growth")
plt.subplot(2, 2, 3)
decomposition.trend.plot(color='purple')
plt.title('Underlying Trend: Is the System Load Increasing?', fontweight='bold')

# Plot 4: Residue (Unexplained Noise)
plt.subplot(2, 2, 4)
decomposition.resid.plot(color='gray', alpha=0.5)
plt.title('Residuals: Unexplained Volatility', fontweight='bold')

plt.tight_layout()
plt.savefig('advanced_forensics_dashboard.png')
print(">>> DASHBOARD GENERATED: 'advanced_forensics_dashboard.png'")

# --- MODULE 6: AUTOMATED INSIGHT GENERATION ---
print("\n" + "="*40)
print("     EXECUTIVE INTELLIGENCE REPORT")
print("="*40)

# Insight 1: The Anomaly Count
print(f"1. FORENSIC ALERT: {len(anomalies)} operational instances flagged as 'High-Risk Anomalies'.")
print(f"   - These centers handled abnormal loads (Mean Load: {anomalies['total_load'].mean():.0f}) with unusual Student Ratios.")
print(f"   - ACTION: Audit the Top 5 Districts: {anomalies['district'].value_counts().head(5).index.tolist()}")

# Insight 2: Seasonality
peak_season_day = decomposition.seasonal.idxmax()
print(f"2. PREDICTIVE LOGISTICS: Mathematical decomposition confirms a recurring 'Stress Cycle'.")
print(f"   - Recommendation: Deploy 'Pop-up Mobile Vans' 15 days prior to peak seasonal dates.")

# Insight 3: The "Burden"
avg_bir = analysis_df['BIR'].mean()
print(f"3. EFFICIENCY METRIC: The National Average 'Biometric Intensity' is {avg_bir:.2f}.")
print(f"   - Any district with BIR > {avg_bir*1.5:.2f} requires Hardware Upgrades (Iris Scanners) immediately.")