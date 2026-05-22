import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go
from huggingface_hub import InferenceClient
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="PJM Forecast & GenAI", page_icon="⚡", layout="wide")

# --- Load Model & Metadata ---
@st.cache_resource
def load_xgb_and_meta():
    try:
        model = joblib.load("models/xgboost_model.pkl")
        with open("models/metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        return model, meta, None
    except Exception as e:
        return None, None, str(e)

xgb_model, meta, load_error = load_xgb_and_meta()
if meta:
    feature_cols = meta.get("feature_columns", [])
    perf = meta.get("model_performance", {}).get("XGBoost", {})
    default_mape = perf.get('MAPE', 1.0)
else:
    feature_cols = []
    perf = {}
    default_mape = 1.0

# --- Helper Functions ---
def add_confidence_intervals(df, mape):
    df['lower'] = df['predicted_consumption'] * (1 - mape/100)
    df['upper'] = df['predicted_consumption'] * (1 + mape/100)
    return df

def create_features_for_prediction(dt, lag_vals):
    hour = dt.hour
    day_of_week = dt.weekday()
    month = dt.month
    year = dt.year
    quarter = (month - 1) // 3 + 1
    is_weekend = 1 if day_of_week >= 5 else 0
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    season = season_map[month]
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    features = {
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'year': year,
        'quarter': quarter,
        'is_weekend': is_weekend,
        'season': season,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'lag_1h': lag_vals.get('lag_1h', 5600),
        'lag_24h': lag_vals.get('lag_24h', 5600),
        'lag_168h': lag_vals.get('lag_168h', 5600),
        'rolling_mean_24h': lag_vals.get('rolling_mean_24h', 5600),
        'rolling_std_24h': lag_vals.get('rolling_std_24h', 1000)
    }
    return pd.DataFrame([features])[feature_cols]

def generate_30_day_forecast(model, start_date, initial_lag, mape_adj=1.0, demand_shift=0):
    forecasts = []
    current_dt = start_date
    recent = [initial_lag['lag_1h']] * 168
    for _ in range(30 * 24):
        lag_vals = {
            'lag_1h': recent[-1] if len(recent) >= 1 else 5600,
            'lag_24h': recent[-24] if len(recent) >= 24 else 5600,
            'lag_168h': recent[-168] if len(recent) >= 168 else 5600,
            'rolling_mean_24h': np.mean(recent[-24:]) if len(recent) >= 24 else 5600,
            'rolling_std_24h': np.std(recent[-24:]) if len(recent) >= 24 else 1000
        }
        features = create_features_for_prediction(current_dt, lag_vals)
        pred = model.predict(features)[0] + demand_shift
        forecasts.append({'datetime': current_dt, 'predicted_consumption': pred})
        recent.append(pred)
        if len(recent) > 168:
            recent = recent[-168:]
        current_dt += timedelta(hours=1)
    df = pd.DataFrame(forecasts)
    df = add_confidence_intervals(df, mape_adj)
    return df

def generate_peak_table(df):
    peak_hours = df.sort_values('predicted_consumption', ascending=False).head(5)
    peak_hours = peak_hours[['datetime', 'predicted_consumption', 'upper', 'lower']]
    peak_hours.columns = ['DateTime', 'Predicted MW', 'Upper Bound', 'Lower Bound']
    return peak_hours

def generate_automated_insights(df, mape):
    max_row = df.loc[df['predicted_consumption'].idxmax()]
    min_row = df.loc[df['predicted_consumption'].idxmin()]
    avg = df['predicted_consumption'].mean()
    return (
        f"• **Peak demand** is expected on **{max_row['datetime'].strftime('%Y-%m-%d %H:%M')}** at **{max_row['predicted_consumption']:.1f} MW**.\n"
        f"• **Lowest demand** is expected on **{min_row['datetime'].strftime('%Y-%m-%d %H:%M')}** at **{min_row['predicted_consumption']:.1f} MW**.\n"
        f"• **Average demand** over 30 days is **{avg:.1f} MW**.\n"
        f"• 95% of predictions are expected to be within ±{mape:.2f}% of the forecasted value."
    )

def generate_report_text(df, perf, mape):
    max_row = df.loc[df['predicted_consumption'].idxmax()]
    min_row = df.loc[df['predicted_consumption'].idxmin()]
    avg = df['predicted_consumption'].mean()
    total = df['predicted_consumption'].sum() / 1000
    lines = [
        "PJM 30-Day Energy Forecast Report",
        "="*35,
        f"Forecast Start: {df['datetime'].iloc[0]}",
        f"Forecast End:   {df['datetime'].iloc[-1]}",
        "",
        f"Peak Demand: {max_row['predicted_consumption']:.1f} MW at {max_row['datetime']}",
        f"Lowest Demand: {min_row['predicted_consumption']:.1f} MW at {min_row['datetime']}",
        f"Average Demand: {avg:.1f} MW",
        f"Total Energy: {total:.1f} GWh",
        f"Model: XGBoost (MAPE: {mape:.2f}%)",
        "",
        "Forecast Table (first 10 rows):",
        df.head(10).to_string(index=False)
    ]
    return "\n".join(lines)

# --- GenAI (Llama-3) for Q&A and Report ---
def ask_hf_llm(question, df, colname='predicted_consumption'):
    HF_TOKEN = st.secrets["hf_token"]["hf_token"]
    client = InferenceClient("meta-llama/Meta-Llama-3-70B-Instruct", token=HF_TOKEN)
    peak = df[colname].max()
    avg = df[colname].mean()
    min_ = df[colname].min()
    context = (
        f"PJM energy data summary:\n"
        f"- Peak: {peak:.1f} MW\n"
        f"- Average: {avg:.1f} MW\n"
        f"- Minimum: {min_:.1f} MW\n"
        f"- Data: {df.head(10).to_csv(index=False)}\n"
    )
    prompt = (
        f"{context}\n"
        f"User question: {question}\n"
        f"Answer in a concise, business-friendly way."
    )
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = client.chat_completion(messages=messages, max_tokens=256, temperature=0.2)
    return response.choices[0].message.content.strip()

def generate_hf_report(df, colname='predicted_consumption'):
    HF_TOKEN = st.secrets["hf_token"]["hf_token"]
    client = InferenceClient("meta-llama/Meta-Llama-3-70B-Instruct", token=HF_TOKEN)
    peak = df[colname].max()
    avg = df[colname].mean()
    min_ = df[colname].min()
    context = (
        f"PJM energy data summary:\n"
        f"- Peak: {peak:.1f} MW\n"
        f"- Average: {avg:.1f} MW\n"
        f"- Minimum: {min_:.1f} MW\n"
        f"- Data: {df.head(10).to_csv(index=False)}\n"
    )
    prompt = (
        f"{context}\n"
        "Generate a detailed business report with:\n"
        "- Executive summary\n"
        "- Key findings\n"
        "- Recommendations for energy managers\n"
        "- Any risks or opportunities\n"
        "Use clear, professional language."
    )
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = client.chat_completion(messages=messages, max_tokens=512, temperature=0.2)
    return response.choices[0].message.content.strip()

# --- Data Loader for Dataset Tab ---
@st.cache_data
def load_data():
    if os.path.exists("PJMW_hourly.csv"):
        df = pd.read_csv("PJMW_hourly.csv")
        if 'Datetime' in df.columns and 'PJMW' in df.columns:
            df = df.rename(columns={'Datetime':'datetime', 'PJMW':'MW'})
        else:
            df.columns = ['datetime', 'MW']
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime', 'MW'])
        df = df.sort_values('datetime')
        return df
    elif os.path.exists("PJMW_MW_Hourly.xlsx"):
        df = pd.read_excel("PJMW_MW_Hourly.xlsx", header=None, names=['datetime','MW'])
        df['datetime'] = pd.to_datetime(df['datetime'].astype(str).str[:15], format='%Y-%m-%d %H%M%S', errors='coerce')
        df['MW'] = pd.to_numeric(df['MW'], errors='coerce')
        df = df.dropna(subset=['datetime', 'MW'])
        df = df.sort_values('datetime')
        return df
    else:
        return None

# --- Tabs UI ---
tab1, tab2 = st.tabs(["🔮 Forecast + GenAI Q&A", "🤖 GenAI Q&A on Dataset"])

with tab1:
    st.markdown("""
    # ⚡ PJM 30-Day Forecast + GenAI Q&A

    - Generate and visualize a 30-day forecast using XGBoost.
    - Scenario analysis, peak table, automated insights, and downloadable reports.
    - **Ask GenAI about the forecast** (business, operational, or analytical questions).
    - Download a GenAI-powered business report.
    ---
    """)
    st.sidebar.header("Forecast Controls (Tab 1)")
    start_date = st.sidebar.date_input("Forecast Start Date", value=datetime.now().date())
    start_dt = datetime.combine(start_date, datetime.min.time())
    init_val = st.sidebar.number_input("Current Consumption (MW)", value=5600.0, step=100.0)
    mape_multiplier = st.sidebar.slider(
        "Adjust Uncertainty (MAPE %)", min_value=0.5, max_value=5.0, value=float(default_mape), step=0.1,
        help="Simulate higher or lower forecast uncertainty"
    )
    demand_shift = st.sidebar.slider(
        "Adjust Demand (MW)", min_value=-1000, max_value=1000, value=0, step=50,
        help="Simulate a one-time increase or decrease in all forecasted values"
    )
    if st.sidebar.button("📈 Generate 30-Day Forecast", use_container_width=True):
        with st.spinner("Generating forecast..."):
            initial_lag = {
                'lag_1h': init_val,
                'lag_24h': init_val,
                'lag_168h': init_val,
                'rolling_mean_24h': init_val,
                'rolling_std_24h': 1000.0
            }
            forecast_df = generate_30_day_forecast(xgb_model, start_dt, initial_lag, mape_multiplier, demand_shift)
            st.session_state['forecast_df'] = forecast_df
            st.success("✅ Forecast generated!")

    if 'forecast_df' in st.session_state:
        forecast_df = st.session_state['forecast_df']
        st.subheader("📈 30-Day Forecast Visualization")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['datetime'], forecast_df['datetime'][::-1]]),
            y=pd.concat([forecast_df['upper'], forecast_df['lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(30,144,255,0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Confidence Interval'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['datetime'],
            y=forecast_df['predicted_consumption'],
            mode='lines',
            name='Predicted Consumption',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title="30-Day Hourly Energy Consumption Forecast (with Confidence Interval)",
            xaxis_title="Date",
            yaxis_title="Consumption (MW)",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 Forecast Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg MW", f"{forecast_df['predicted_consumption'].mean():.1f}")
        with col2:
            st.metric("Peak MW", f"{forecast_df['predicted_consumption'].max():.1f}")
        with col3:
            st.metric("Min MW", f"{forecast_df['predicted_consumption'].min():.1f}")
        with col4:
            st.metric("Total GWh", f"{forecast_df['predicted_consumption'].sum()/1000:.1f}")

        st.markdown("### 🔥 Peak Demand Table")
        peak_table = generate_peak_table(forecast_df)
        st.table(peak_table.style.format({'Predicted MW': '{:.1f}', 'Upper Bound': '{:.1f}', 'Lower Bound': '{:.1f}'}))

        st.markdown("### 💡 Automated Insights")
        st.info(generate_automated_insights(forecast_df, mape_multiplier))

        report_text = generate_report_text(forecast_df, perf, mape_multiplier)
        st.download_button(
            label="📄 Download Forecast Report (TXT)",
            data=report_text,
            file_name=f"pjm_30day_report_{start_date}.txt",
            mime="text/plain"
        )

        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"pjm_30day_forecast_{start_date}.csv",
            mime="text/csv"
        )

        # --- GenAI Q&A for Forecast ---
        st.markdown("### 💬 Ask GenAI about the 30-day forecast")
        user_question = st.text_input("Ask a question about the forecast (e.g., 'What is the highest demand day?')", key="forecast_qa")
        if st.button("Ask GenAI", key="btn_forecast_qa"):
            if user_question.strip():
                with st.spinner("GenAI is thinking..."):
                    answer = ask_hf_llm(user_question, forecast_df, colname='predicted_consumption')
                st.success(answer)
            else:
                st.info("Please enter a question.")

        # --- GenAI Automated Report ---
        st.markdown("### 🤖 GenAI-Powered Automated Report")
        if st.button("Download GenAI Report (TXT)", key="btn_forecast_report"):
            with st.spinner("GenAI is generating your report..."):
                report_txt = generate_hf_report(forecast_df, colname='predicted_consumption')
            st.download_button(
                label="📥 Download GenAI Report",
                data=report_txt,
                file_name=f"pjm_genai_report_{start_date}.txt",
                mime="text/plain"
            )
            st.success("GenAI report generated and ready to download!")
    else:
        st.info("Set the start date and initial value, then click 'Generate 30-Day Forecast'.")

with tab2:
    st.markdown("""
    # 🤖 GenAI Q&A on Full Dataset

    - Ask GenAI about any aspect of your full PJM dataset (not just the forecast).
    - Download a business-style report for the dataset.
    - See quick visualizations and stats.

    ---
    **Try asking GenAI questions like:**
    - What is the highest demand day in the data?
    - What is the average demand for July?
    - Are there any seasonal trends or unusual patterns?
    - How does demand change on weekends vs weekdays?
    - When was the lowest demand recorded?
    - Generate a summary report for the last year.
    ---
    """)
    df = load_data()
    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head(48))
        st.subheader("Quick Stats")
        st.write(df['MW'].describe())
        st.subheader("Monthly Average Demand")
        df['month'] = df['datetime'].dt.month
        st.bar_chart(df.groupby('month')['MW'].mean())

        st.markdown("### 💬 Ask GenAI about the dataset")
        user_question2 = st.text_input("Ask a question about the dataset (e.g., 'What is the highest demand day?')", key="dataset_qa")
        if st.button("Ask GenAI", key="btn_dataset_qa"):
            if user_question2.strip():
                with st.spinner("GenAI is thinking..."):
                    answer = ask_hf_llm(user_question2, df, colname='MW')
                st.success(answer)
            else:
                st.info("Please enter a question.")

        st.markdown("### 🤖 GenAI-Powered Automated Report")
        if st.button("Download GenAI Dataset Report (TXT)", key="btn_dataset_report"):
            with st.spinner("GenAI is generating your report..."):
                report_txt = generate_hf_report(df, colname='MW')
            st.download_button(
                label="📥 Download GenAI Report",
                data=report_txt,
                file_name=f"pjm_genai_dataset_report.txt",
                mime="text/plain"
            )
            st.success("GenAI report generated and ready to download!")
    else:
        st.warning("No PJM data file found. Please upload PJMW_hourly.csv or PJMW_MW_Hourly.xlsx.")

st.markdown("---")
st.markdown("**Powered by XGBoost + Hugging Face Llama-3 GenAI | Data Science Expert**")
