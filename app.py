# app.py

import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="💳 ATM Card Validator",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 12px;
        }
        .stButton>button {
            background-color: #0099ff;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stDownloadButton>button {
            background-color: #00c853;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .title {
            font-size: 32px;
            font-weight: bold;
            color: #003366;
        }
        .subtitle {
            font-size: 18px;
            color: #333333;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<p class="title">💳 ATM Card Detection Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a CSV file with ATM card data to classify cards as <b style="color:green;">Real</b> or <b style="color:red;">Fake</b>.</p>', unsafe_allow_html=True)

# Load trained model
try:
    model = joblib.load("atm_card_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'atm_card_model.pkl' is in the same directory.")
    st.stop()

# File uploader section
st.markdown("### 📁 Upload ATM Card Test Data")
uploaded_file = st.file_uploader("Choose a CSV file...", type=["csv"])

# Process uploaded file
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Required columns
        required_cols = ["Mag_Stripe_Integrity", "Chip_Auth", "Hidden_Code_Present", "Slot_Physical_Tampering"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ CSV must contain the following columns: {required_cols}")
        else:
            # Predict
            predictions = model.predict(df[required_cols])
            df["Prediction"] = ["🟢 Real" if p == 0 else "🔴 Fake" for p in predictions]

            st.success("✅ Predictions completed successfully!")
            st.markdown("### 📊 Results")
            st.dataframe(df.style.highlight_max(axis=0, color="lightgreen"))

            # Download button
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv, "predicted_results.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("📝 Please upload a CSV file with valid input data to begin prediction.")

st.markdown('</div>', unsafe_allow_html=True)
