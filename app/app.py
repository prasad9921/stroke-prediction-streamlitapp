import streamlit as st
import pandas as pd
import joblib
import json
import os

# Import modules
import data_insights
import model_insights
import inference

# Page Config
st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Reduce top padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    /* Style Tabs to look like large buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        padding: 10px 30px;
        font-size: 20px; /* Larger text */
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0068c9 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load resources (cached)
@st.cache_resource
def load_resources():
    try:
        # Load Data for Insights
        df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
        
        # Load Metadata
        with open('model_assets/model_metadata.json', 'r') as f:
            metadata = json.load(f)
            
        # Load Preprocessor
        preprocessor = joblib.load('model_assets/preprocessor.joblib')
        
        return df, metadata, preprocessor
    except FileNotFoundError as e:
        st.error(f"Missing File: {e}. Please ensure dataset and model files are in the directory.")
        return None, None, None

def main():
    st.title("🧠 Stroke Prediction & Analysis System")
    st.markdown("### AI-Powered Health Risk Assessment")
    st.markdown("---")

    # Load resources
    df, metadata, preprocessor = load_resources()

    if df is not None:
        # Create Tabs
        tab1, tab2, tab3 = st.tabs(["📊 DATA INSIGHTS", "🤖 MODEL INSIGHTS", "🩺 INFERENCE LAB"])

        with tab1:
            data_insights.show(df)
        
        with tab2:
            model_insights.show(metadata)
            
        with tab3:
            inference.show(preprocessor, metadata)

if __name__ == "__main__":
    main()