import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

# --- 1. ENTERPRISE CONFIGURATION ---
st.set_page_config(
    page_title="UIDAI: National Command Center",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- THE "SOVEREIGN GLASS" CSS (REFINED) ---
st.markdown("""
<style>
    /* Global Reset */
    .stApp { background-color: #f4f6f9; font-family: 'Inter', 'Segoe UI', sans-serif; }
    
    /* Neomorphic Metric Cards */
    div[data-testid="stMetric"] { 
        background-color: #ffffff; 
        border: 1px solid #eef2f6; 
        padding: 20px; 
        border-radius: 16px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); 
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-color: #cbd5e1;
    }
    
    /* HEADERS */
    h1 { color: #1e293b; font-weight: 900; letter-spacing: -1px; }
    h2, h3 { color: #334155; font-weight: 700; }
    
    /* --- TAB STYLING --- */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 24px; 
        background-color: transparent; 
        padding: 10px 0px; 
        margin-bottom: 20px;
    }
    
    /* Individual Tab Styling */
    .stTabs [data-baseweb="tab"] { 
        height: 45px; 
        border-radius: 30px; 
        color: #64748b; 
        font-weight: 600; 
        border: 1px solid transparent;
        background-color: #ffffff;
        padding: 0 24px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        transition: all 0.2s;
    }
    
    /* Selected Tab */
    .stTabs [aria-selected="true"] { 
        background-color: #0f172a; 
        color: #ffffff; 
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.3); 
        border: 1px solid #0f172a;
    }
    
    /* Simulator Container Styling */
    .simulator-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin-top: 30px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA LAYER ---
@st.cache_data
def load_data():
    try:
        plan = pd.read_csv('artifacts/final_scientific_plan.csv')
        prog = pd.read_csv('artifacts/prognosis_artifact.csv')
        causal = pd.read_csv('artifacts/causal_artifact.csv')
        
        # GEO-ENRICHMENT
        coords = {
            'Pune': [18.5204, 73.8567], 'Thane': [19.2183, 72.9781], 'Nashik': [19.9975, 73.7898],
            'Ahmedabad': [23.0225, 72.5714], 'Jaipur': [26.9124, 75.7873], 'Bengaluru': [12.9716, 77.5946],
            'Lucknow': [26.8467, 80.9462], 'Hyderabad': [17.3850, 78.4867], 'Mumbai': [19.0760, 72.8777],
            'Delhi': [28.7041, 77.1025]
        }
        
        def get_lat(d): return coords.get(d, [20.5937, 78.9629])[0] + np.random.normal(0, 0.05)
        def get_lon(d): return coords.get(d, [20.5937, 78.9629])[1] + np.random.normal(0, 0.05)
        
        plan['lat'] = plan['district'].apply(get_lat)
        plan['lon'] = plan['district'].apply(get_lon)
        plan['plot_height'] = (plan['total_load'] / plan['total_load'].max()) * 100000 
        
        return plan, prog, causal
    except Exception as e:
        st.error(f"System Offline: Artifacts missing. Please run 'run_pipeline.py'. Error: {e}")
        return None, None, None

plan_df, prog_df, causal_df = load_data()
if plan_df is None: st.stop()

# --- 3. HEADER ---
c1, c2 = st.columns([3, 1])
with c1:
    st.title("🇮🇳 UIDAI: Operational Intelligence Grid")
    st.markdown("**System Status:** :green[● Online] &nbsp; | &nbsp; **Simulation:** :blue[● M/G/k Active]")
with c2:
    st.caption("SECURE CONNECTION")
    st.markdown("`ID: 2026-ALPHA-01`")
st.markdown("---")

# --- 4. KPIs ---
k1, k2, k3, k4 = st.columns(4)
total_stress_red = causal_df['ROI'].sum() if causal_df is not None else 0
daily_savings = int(total_stress_red * 500)
crit_districts = len(plan_df[plan_df['Stress_Level_Percent'] > 100])
avg_wait = int(plan_df['Est_Wait_Time_Mins'].mean())

k1.metric("Critical Nodes", f"{crit_districts}", "Load > 100%", delta_color="inverse")
k2.metric("Network Latency", f"{avg_wait} min", "Avg Wait Time", delta_color="inverse")
k3.metric("Economic Recovery", f"₹ {daily_savings:,}", "Daily Value Add")
k4.metric("Active Sensors", f"{len(plan_df)}", "Biometric Units")

# --- 5. TABS ---
st.write("") 
tab_ops, tab_prog, tab_causal, tab_intel = st.tabs(["📍 Command Map", "🔮 Prognostic Vision", "🧪 Causal Lab", "📑 Executive Brief"])

# --- TAB 1: WAR ROOM ---
with tab_ops:
    
    # 3D MAP SECTION
    st.subheader("Geospatial Load Digital Twin")
    
    # LEGEND
    st.markdown("""
    <div style="background-color: white; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 15px; display: flex; gap: 20px; align-items: center;">
        <span style="font-weight: 600; font-size: 14px; color: #64748b;">VISUAL LEGEND:</span>
        <span style="color: #ef4444; font-weight: 700; font-size: 14px;">🔴 Critical (Wait > 1 Hour)</span>
        <span style="color: #10b981; font-weight: 700; font-size: 14px;">🟢 Stable (Wait < 1 Hour)</span>
        <span style="color: #64748b; font-size: 12px;">(Height represents Total Load)</span>
    </div>
    """, unsafe_allow_html=True)

    layer = pdk.Layer(
        "ColumnLayer",
        data=plan_df,
        get_position=["lon", "lat"],
        get_elevation="plot_height",
        elevation_scale=1,
        radius=5000,
        get_fill_color=["Stress_Level_Percent > 100 ? [239, 68, 68] : [16, 185, 129]"],
        pickable=True,
        auto_highlight=True,
    )
    
    view_state = pdk.ViewState(latitude=20.5937, longitude=78.9629, zoom=4, pitch=45)
    
    # --- FIX 1: REPLACED MAPBOX WITH CARTO (OPEN SOURCE) TO FIX FLOATING DOTS ---
    r = pdk.Deck(
        layers=[layer], 
        initial_view_state=view_state, 
        tooltip={"text": "District: {district}\nLoad: {total_load}\nStress: {Stress_Level_Percent}%"}, 
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
    )
    st.pydeck_chart(r)

    # --- SIMULATOR SECTION ---
    st.markdown("---")
    st.subheader("🛠️ Diagnostics & Simulation Lab")
    st.markdown("Use this module to visualize how **Wait Times** react to hardware intervention.")

    @st.fragment 
    def run_simulation():
        with st.container():
            c_sim_1, c_sim_2 = st.columns([1, 2])
            
            with c_sim_1:
                st.markdown("#### 🎛️ Control Panel")
                added_kits = st.slider("Deploy Strategic Reserves (Kits)", 0, 50000, 0, 1000)
                
                sim_df = plan_df.copy()
                total_load_national = sim_df['total_load'].sum()
                sim_df['New_Kits'] = (sim_df['total_load'] / total_load_national) * added_kits
                decay_factor = 5 / (5 + sim_df['New_Kits'])
                sim_df['Simulated_Wait'] = sim_df['Est_Wait_Time_Mins'] * decay_factor
                
                new_avg_wait = int(sim_df['Simulated_Wait'].mean())
                wait_delta = new_avg_wait - avg_wait
                
                st.metric("Projected Avg Wait", f"{new_avg_wait} min", f"{wait_delta} min", delta_color="inverse")
                if new_avg_wait < 60: st.success("✅ GRID STABILIZED")
                else: st.warning("⚠️ GRID UNSTABLE")

            with c_sim_2:
                sim_df['Status'] = sim_df['Simulated_Wait'].apply(lambda x: 'Critical' if x > 60 else 'Stable')
                
                # --- FIX 2: REMOVED TEXT LABELS TO FIX GREY SMUDGE/CLUTTER ---
                fig_quad = px.scatter(
                    sim_df, 
                    x='total_load', 
                    y='Simulated_Wait', 
                    size='Optimal_Counters', 
                    color='Status',
                    color_discrete_map={'Critical': '#ef4444', 'Stable': '#10b981'},
                    hover_name='district',
                    log_x=True,
                    title=f"Impact Analysis: +{added_kits} Kits Deployed",
                    labels={'total_load': 'Daily Citizen Load (Log Scale)', 'Simulated_Wait': 'Est. Wait Time (Minutes)'},
                    template="plotly_white",
                    height=350
                )
                
                fig_quad.add_hline(y=60, line_dash="dash", line_color="green", annotation_text="1 Hour Target")
                
                # --- FIX 3: ENSURED CONTAINER WIDTH TO FIX DISTORTION ---
                st.plotly_chart(fig_quad, use_container_width=True) 

    run_simulation()

# --- TAB 2: PROGNOSIS ---
with tab_prog:
    st.subheader("Deep Temporal Forecasting (LSTM)")
    if prog_df is not None:
        target_dist = prog_df['District'].iloc[0]
        fig_prog = px.line(prog_df, x='Day', y='Load', color='Type', markers=True,
                          color_discrete_map={'Historical': '#94a3b8', 'AI Forecast': '#f59e0b'},
                          title=f"Predictive Load: {target_dist}", template="plotly_white")
        fig_prog.add_vline(x=0, line_dash="dash", annotation_text="Today")
        fig_prog.update_layout(height=400)
        st.plotly_chart(fig_prog, use_container_width=True)

# --- TAB 3: CAUSAL ---
with tab_causal:
    st.subheader("Double Machine Learning (DML)")
    if causal_df is not None:
        fig_causal = go.Figure()
        fig_causal.add_trace(go.Scatter(x=pd.concat([causal_df['Date'], causal_df['Date'][::-1]]),
            y=pd.concat([causal_df['Baseline'], causal_df['Optimized'][::-1]]),
            fill='toself', fillcolor='rgba(16, 185, 129, 0.1)', line=dict(color='rgba(0,0,0,0)'), name='Efficiency Gain'))
        fig_causal.add_trace(go.Scatter(x=causal_df['Date'], y=causal_df['Baseline'], mode='lines', name='Status Quo', line=dict(color='#ef4444', width=2)))
        fig_causal.add_trace(go.Scatter(x=causal_df['Date'], y=causal_df['Optimized'], mode='lines', name='With Intervention', line=dict(color='#10b981', width=2, dash='dash')))
        fig_causal.update_layout(title="ROI Verification", template="plotly_white", height=450)
        st.plotly_chart(fig_causal, use_container_width=True)

# --- TAB 4: BRIEF ---
with tab_intel:
    st.subheader("Download Strategic Directive")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(plan_df[['district', 'total_load', 'Stress_Level_Percent', 'Est_Wait_Time_Mins']].head(50), height=400, use_container_width=True)

    with c2:
        st.markdown("### Export Protocols")
        st.write("Generate encrypted CSV.")
        csv = plan_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Plan", csv, "uidai_strategy_2026.csv", "text/csv", type="primary")