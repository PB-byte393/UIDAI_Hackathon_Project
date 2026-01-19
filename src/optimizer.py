import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
TARGET_UTILIZATION = 0.85  
WORKING_HOURS = 8
WORKING_MINUTES = WORKING_HOURS * 60

def optimize_network():
    print(">>> [INIT] Initializing Operations Research Engine...")

    def load_shards(pattern):
        files = glob.glob(pattern)
        print(f"Loading {len(files)} files from {pattern}...")
        df_list = []
        for f in files:
            try:
                cols = pd.read_csv(f, nrows=1).columns.tolist()
                age_cols = [c for c in cols if 'age' in c]
                keep = ['district'] + age_cols
                temp = pd.read_csv(f, usecols=keep)
                temp = temp.groupby('district')[age_cols].sum().reset_index()
                df_list.append(temp)
            except: continue
        if not df_list: return pd.DataFrame()
        return pd.concat(df_list, ignore_index=True)

    bio_df = load_shards('data/api_data_aadhar_biometric_*.csv')
    demo_df = load_shards('data/api_data_aadhar_demographic_*.csv')

    full = pd.merge(bio_df, demo_df, on='district', suffixes=('_bio', '_demo'), how='outer').fillna(0)

    print(">>> [PROCESSING] Aggregating District Loads...")
    
    bio_cols = [c for c in full.columns if '_bio' in c or 'bio_' in c]
    demo_cols = [c for c in full.columns if '_demo' in c or 'demo_' in c]
    
    full['Biometric_Load'] = full[bio_cols].sum(axis=1)
    full['Demographic_Load'] = full[demo_cols].sum(axis=1)
    full['total_load'] = full['Biometric_Load'] + full['Demographic_Load']
    
    # --- PHYSICS ENGINE: SERVICE RATES ---
    # Avg Demo Time: 5 mins | Avg Bio Time: 15 mins
    full['Avg_Service_Min'] = (
        (full['Biometric_Load'] * 15) + (full['Demographic_Load'] * 5)
    ) / (full['total_load'] + 1)
    
    full['Avg_Service_Min'] = full['Avg_Service_Min'].replace(0, 10)

    print(">>> [OPTIMIZATION] Solving M/G/k Queue Equations...")
    
    # 1. Capacity per Kit
    full['Daily_Capacity_Per_Kit'] = WORKING_MINUTES / full['Avg_Service_Min']
    
    # 2. Current Utilization (Rho) assuming 5 baseline kits
    current_kits = 5
    full['Current_Rho'] = full['total_load'] / (full['Daily_Capacity_Per_Kit'] * current_kits)
    
    # 3. Stress Percentage
    full['Stress_Level_Percent'] = full['Current_Rho'] * 100

    # 4. OPTIMAL KITS (To reach Target Utilization)
    full['Optimal_Counters'] = np.ceil(full['total_load'] / (full['Daily_Capacity_Per_Kit'] * TARGET_UTILIZATION))
    
    # --- UPGRADE: KINGMAN'S FORMULA FOR WAIT TIME ---
    # Wait_Time = (Rho / (1-Rho)) * (Variation / 2) * Service_Time
    # We assume 'Variation' (Coefficient of Variation) is 1 (Exponential randomness)
    # We clip Rho at 0.99 to prevent infinite wait time in the calculation
    
    rho_clipped = full['Current_Rho'].clip(upper=0.99)
    term_1 = (rho_clipped / (1 - rho_clipped))
    term_2 = 0.5 # Variation factor (CV^2 + 1)/2, assuming Poisson arrivals
    
    full['Est_Wait_Time_Mins'] = term_1 * term_2 * full['Avg_Service_Min']
    
    # If Rho was actually > 1, the wait time is effectively "Infinite" (set to 8 hours for display cap)
    full.loc[full['Current_Rho'] > 1, 'Est_Wait_Time_Mins'] = 480 

    # --- OUTPUT GENERATION ---
    final_plan = full[['district', 'total_load', 'Avg_Service_Min', 'Stress_Level_Percent', 'Est_Wait_Time_Mins', 'Optimal_Counters']].copy()
    final_plan = final_plan.sort_values('Stress_Level_Percent', ascending=False)
    final_plan = final_plan[final_plan['total_load'] > 50]
    
    final_plan.to_csv('artifacts/final_scientific_plan.csv', index=False)
    
    # VISUALIZATION
    if not final_plan.empty:
        plt.figure(figsize=(10, 6))
        # Plot Wait Time instead of just Stress
        sns.scatterplot(data=final_plan, x='total_load', y='Est_Wait_Time_Mins', 
                        hue='Stress_Level_Percent', size='Optimal_Counters', sizes=(20, 200), palette='magma')
        plt.title('Queueing Physics: Est. Wait Time vs Load')
        plt.xlabel('Daily Load (Log Scale)')
        plt.ylabel('Est. Citizen Wait Time (Minutes)')
        plt.xscale('log')
        plt.axhline(60, color='red', linestyle='--', label='1 Hour Threshold') # 1 Hour Threshold
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('images/queueing_simulation.png')
        print(">>> [VISUAL] Generated 'queueing_simulation.png' with Kingman's Approx.")
    
    print("\n============================================================")
    print("   SCIENTIFIC RESOURCE ALLOCATION (M/G/k + KINGMAN)")
    print("============================================================")
    if not final_plan.empty:
        print(final_plan.head(5))
        avg_wait = final_plan['Est_Wait_Time_Mins'].mean()
        print(f"\n>>> [INSIGHT] Average Network Wait Time: {avg_wait:.1f} Minutes")

if __name__ == "__main__":
    optimize_network()