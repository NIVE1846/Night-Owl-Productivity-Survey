import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import re
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

st.set_page_config(page_title="Night Owl Survey", page_icon="🦉", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    .main > div {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    h1 {
        color: #667eea;
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        border: none;
        font-size: 1.1rem;
        width: 100%;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    
    h2 {
        color: #764ba2;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #667eea;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

CSV_FILE = "survey_responses.csv"

def validate_email(email):
    return re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email)

def save_response(data):
    df = pd.DataFrame([data])
    try:
        if os.path.exists(CSV_FILE):
            df.to_csv(CSV_FILE, mode='a', header=False, index=False)
        else:
            df.to_csv(CSV_FILE, index=False)
    except:
        # Fallback: use session state if file system not writable
        if 'responses' not in st.session_state:
            st.session_state.responses = []
        st.session_state.responses.append(data)

def load_data():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    elif 'responses' in st.session_state and st.session_state.responses:
        return pd.DataFrame(st.session_state.responses)
    return None

page = st.sidebar.radio("Navigate", ["🦉 Survey", "📊 Analysis"])

if page == "🦉 Survey":
    st.markdown('<h1>🦉 Night Owl Productivity Survey</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Help us understand productivity patterns across different chronotypes</p>', unsafe_allow_html=True)
    
    with st.form("survey_form"):
        st.subheader("📋 Demographics")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name*")
            age = st.number_input("Age*", min_value=15, max_value=100, step=1, value=25)
        with col2:
            email = st.text_input("Email*")
            occupation = st.selectbox("Occupation*", ["Student", "Software Developer", "Designer", "Manager", "Teacher", "Healthcare Worker", "Entrepreneur", "Freelancer", "Other"])
        
        st.subheader("⏰ Work & Activity Timing")
        work_hours = st.select_slider("When are you most productive?*", 
                                       options=["Early Morning (5-8 AM)", "Morning (8-11 AM)", "Midday (11 AM-2 PM)", 
                                               "Afternoon (2-5 PM)", "Evening (5-9 PM)", "Night (9 PM-12 AM)", "Late Night (12-5 AM)"])
        chronotype = st.radio("You are:*", ["Early Bird 🌅", "Night Owl 🦉", "Somewhere in Between"])
        
        st.subheader("😴 Sleep Patterns")
        col3, col4 = st.columns(2)
        with col3:
            sleep_time = st.time_input("Usual bedtime*")
            sleep_duration = st.slider("Sleep duration (hours)*", 3, 12, 7)
        with col4:
            wake_time = st.time_input("Usual wake time*")
            sleep_quality = st.select_slider("Sleep quality*", options=["Very Poor", "Poor", "Fair", "Good", "Excellent"])
        
        st.subheader("📱 Digital Habits")
        device_usage = st.slider("Device usage during work (hours)*", 0, 16, 6)
        social_media = st.slider("Social media during work (hours)*", 0, 10, 2)
        distraction_level = st.select_slider("How easily distracted?*", 
                                             options=["Not at all", "Slightly", "Moderately", "Very", "Extremely"])
        
        st.subheader("🎯 Productivity & Focus")
        col5, col6 = st.columns(2)
        with col5:
            productivity = st.slider("Productivity (1-10)*", 1, 10, 5)
            focus_duration = st.slider("Max focus time (minutes)*", 5, 240, 60, step=5)
        with col6:
            stress_level = st.select_slider("Stress level*", options=["Very Low", "Low", "Moderate", "High", "Very High"])
            energy_pattern = st.radio("Energy peaks during:*", ["Morning", "Afternoon", "Evening", "Night"])
        
        next_day_fatigue = st.radio("Next-day fatigue?*", ["Rarely", "Sometimes", "Often", "Always"])
        
        submitted = st.form_submit_button("🚀 Submit Survey")
        
        if submitted:
            if not name or not email:
                st.error("Please fill all required fields!")
            elif not validate_email(email):
                st.error("Invalid email address!")
            else:
                response = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "name": name, "email": email, "age": age, "occupation": occupation,
                    "work_hours": work_hours, "chronotype": chronotype,
                    "sleep_time": sleep_time.strftime("%H:%M"), "wake_time": wake_time.strftime("%H:%M"),
                    "sleep_duration": sleep_duration, "sleep_quality": sleep_quality,
                    "device_usage": device_usage, "social_media": social_media,
                    "distraction_level": distraction_level, "productivity": productivity,
                    "focus_duration": focus_duration, "stress_level": stress_level,
                    "energy_pattern": energy_pattern, "next_day_fatigue": next_day_fatigue
                }
                save_response(response)
                st.success("✅ Response recorded!")
                st.balloons()

else:
    st.title("📊 Analysis Dashboard")
    df = load_data()
    
    if df is None or len(df) == 0:
        st.warning("No data yet!")
    else:
        st.success(f"Total Responses: {len(df)}")
        
        # Advanced data preprocessing
        df['productivity'] = pd.to_numeric(df['productivity'], errors='coerce')
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['device_usage'] = pd.to_numeric(df['device_usage'], errors='coerce')
        df['social_media'] = pd.to_numeric(df['social_media'], errors='coerce')
        df['sleep_duration'] = pd.to_numeric(df['sleep_duration'], errors='coerce')
        df['focus_duration'] = pd.to_numeric(df['focus_duration'], errors='coerce')
        
        # Feature engineering
        df['distraction_score'] = df['social_media'] / (df['focus_duration'] / 60 + 1)
        df['sleep_efficiency'] = df['productivity'] / df['sleep_duration']
        df['digital_dependency'] = df['device_usage'] + df['social_media']
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🦉 Chronotype", "📈 Correlations", "🤖 ML Insights", "💾 Data"])
        
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Productivity", f"{df['productivity'].mean():.1f}/10", 
                       delta=f"{df['productivity'].std():.1f} σ")
            col2.metric("Avg Sleep", f"{df['sleep_duration'].mean():.1f}h",
                       delta=f"{df['sleep_duration'].std():.1f} σ")
            col3.metric("Avg Device Use", f"{df['device_usage'].mean():.1f}h",
                       delta=f"{df['device_usage'].std():.1f} σ")
            col4.metric("Avg Focus", f"{df['focus_duration'].mean():.0f}min",
                       delta=f"{df['focus_duration'].std():.0f} σ")
            
            st.subheader("📊 Distribution Analysis")
            col5, col6 = st.columns(2)
            with col5:
                fig1 = px.violin(df, y='productivity', x='chronotype', box=True, points='all',
                               title="Productivity Distribution by Chronotype",
                               color='chronotype')
                st.plotly_chart(fig1, use_container_width=True)
            with col6:
                fig2 = px.box(df, x='occupation', y='productivity', color='chronotype',
                            title="Productivity by Occupation & Chronotype")
                fig2.update_xaxis(tickangle=45)
                st.plotly_chart(fig2, use_container_width=True)
            
            st.subheader("🎯 Key Performance Indicators")
            col7, col8 = st.columns(2)
            with col7:
                fig3 = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=df['productivity'].mean(),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Overall Productivity Score"},
                    delta={'reference': 5},
                    gauge={'axis': {'range': [None, 10]},
                           'bar': {'color': "#667eea"},
                           'steps': [
                               {'range': [0, 3], 'color': "lightgray"},
                               {'range': [3, 7], 'color': "gray"},
                               {'range': [7, 10], 'color': "darkgray"}],
                           'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 8}}))
                st.plotly_chart(fig3, use_container_width=True)
            with col8:
                fig4 = px.sunburst(df, path=['chronotype', 'occupation'], values='productivity',
                                 title="Productivity Hierarchy")
                st.plotly_chart(fig4, use_container_width=True)
        
        with tab2:
            st.subheader("🦉 Statistical Comparison: Night Owl vs Early Bird")
            
            night_owls = df[df['chronotype'] == 'Night Owl 🦉']
            early_birds = df[df['chronotype'] == 'Early Bird 🌅']
            
            if len(night_owls) > 0 and len(early_birds) > 0:
                # T-test for productivity
                t_stat, p_value = stats.ttest_ind(night_owls['productivity'].dropna(), 
                                                  early_birds['productivity'].dropna())
                
                col1, col2, col3 = st.columns(3)
                col1.metric("T-Statistic", f"{t_stat:.3f}")
                col2.metric("P-Value", f"{p_value:.4f}")
                col3.metric("Significant?", "Yes ✅" if p_value < 0.05 else "No ❌")
                
                st.info(f"**Interpretation:** {'Significant difference' if p_value < 0.05 else 'No significant difference'} in productivity between chronotypes (α=0.05)")
            
            stats_df = df.groupby('chronotype').agg({
                'productivity': ['mean', 'median', 'std'],
                'sleep_duration': ['mean', 'median', 'std'],
                'device_usage': ['mean', 'median', 'std'],
                'focus_duration': ['mean', 'median', 'std'],
                'distraction_score': ['mean', 'median', 'std']
            }).round(2)
            
            st.dataframe(stats_df, use_container_width=True)
            
            col4, col5 = st.columns(2)
            with col4:
                fig5 = px.scatter_3d(df, x='sleep_duration', y='device_usage', z='productivity',
                                   color='chronotype', size='focus_duration',
                                   title="3D Productivity Analysis")
                st.plotly_chart(fig5, use_container_width=True)
            with col5:
                fig6 = px.parallel_coordinates(df, 
                    dimensions=['productivity', 'sleep_duration', 'device_usage', 'focus_duration'],
                    color='productivity', title="Parallel Coordinates Plot")
                st.plotly_chart(fig6, use_container_width=True)
        
        with tab3:
            st.subheader("📈 Advanced Correlation Analysis")
            
            numeric_cols = ['productivity', 'sleep_duration', 'device_usage', 'social_media', 
                          'focus_duration', 'age', 'distraction_score', 'digital_dependency']
            corr = df[numeric_cols].corr()
            
            # Correlation heatmap with annotations
            fig7 = px.imshow(corr, text_auto='.2f', aspect='auto',
                           color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                           title="Pearson Correlation Matrix")
            st.plotly_chart(fig7, use_container_width=True)
            
            # Regression analysis
            st.subheader("🔍 Regression Analysis")
            col6, col7 = st.columns(2)
            with col6:
                fig8 = px.scatter(df, x='sleep_duration', y='productivity', color='chronotype',
                                trendline='ols', trendline_scope='overall',
                                title="Sleep Duration vs Productivity (OLS Regression)")
                st.plotly_chart(fig8, use_container_width=True)
                
                # Calculate R-squared
                from sklearn.linear_model import LinearRegression
                X = df[['sleep_duration']].dropna()
                y = df.loc[X.index, 'productivity']
                model = LinearRegression().fit(X, y)
                r2 = model.score(X, y)
                st.metric("R² Score", f"{r2:.3f}")
            
            with col7:
                fig9 = px.scatter(df, x='distraction_score', y='productivity', color='chronotype',
                                trendline='lowess', title="Distraction Score vs Productivity (LOWESS)")
                st.plotly_chart(fig9, use_container_width=True)
            
            # Pairplot alternative
            st.subheader("📊 Multi-variable Relationships")
            fig10 = px.scatter_matrix(df[['productivity', 'sleep_duration', 'device_usage', 'focus_duration']],
                                    dimensions=['productivity', 'sleep_duration', 'device_usage', 'focus_duration'],
                                    color=df['chronotype'], title="Scatter Matrix")
            fig10.update_traces(diagonal_visible=False)
            st.plotly_chart(fig10, use_container_width=True)
        
        with tab4:
            st.subheader("🤖 Machine Learning Insights")
            
            # K-Means Clustering
            st.markdown("### 🎯 Productivity Clusters")
            features = df[['productivity', 'sleep_duration', 'device_usage', 'focus_duration']].dropna()
            
            if len(features) >= 3:
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(features)
                
                df_clustered = features.copy()
                df_clustered['Cluster'] = clusters
                df_clustered['Cluster'] = df_clustered['Cluster'].map({0: 'High Performers', 1: 'Moderate', 2: 'Low Performers'})
                
                fig11 = px.scatter_3d(df_clustered, x='productivity', y='sleep_duration', z='device_usage',
                                    color='Cluster', size='focus_duration',
                                    title="K-Means Clustering (k=3)")
                st.plotly_chart(fig11, use_container_width=True)
                
                # Cluster statistics
                cluster_stats = df_clustered.groupby('Cluster').agg({
                    'productivity': ['mean', 'count'],
                    'sleep_duration': 'mean',
                    'device_usage': 'mean',
                    'focus_duration': 'mean'
                }).round(2)
                st.dataframe(cluster_stats, use_container_width=True)
            
            # Feature importance (correlation-based)
            st.markdown("### 📊 Feature Importance for Productivity")
            feature_corr = df[numeric_cols].corr()['productivity'].drop('productivity').abs().sort_values(ascending=False)
            fig12 = px.bar(x=feature_corr.values, y=feature_corr.index, orientation='h',
                         title="Feature Correlation with Productivity",
                         labels={'x': 'Absolute Correlation', 'y': 'Feature'})
            st.plotly_chart(fig12, use_container_width=True)
            
            # Statistical summary
            st.markdown("### 📈 Statistical Summary")
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
        
        with tab5:
            st.subheader("💾 Raw Data Export")
            st.dataframe(df, use_container_width=True)
            
            col_export1, col_export2 = st.columns(2)
            with col_export1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv, "survey_data.csv", "text/csv")
            with col_export2:
                json_data = df.to_json(orient='records', indent=2)
                st.download_button("📥 Download JSON", json_data, "survey_data.json", "application/json")

