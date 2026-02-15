import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show(df):
    st.header("Exploratory Data Analysis")
    
    # New Selection/Upload Row
    data_source = st.radio("Data Source:", ["Use Default Healthcare Dataset", "Upload Healthcare data CSV"], horizontal=True)
    
    if data_source == "Upload Healthcare data CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        st.warning("⚠️ **Disclaimer:** Make sure you upload a proper dataset - Stroke Prediction Dataset from Kaggle, else columns will mismatch and rest of the functionalities won't work")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("Custom data loaded successfully!")
    
    st.markdown("---")
    
    # Quick Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", df.shape[0])
    c2.metric("Features", df.shape[1])
    c3.metric("Stroke Cases", df['stroke'].sum())
    c4.metric("Imbalance Ratio", f"1:{round(len(df)/df['stroke'].sum(), 1)}")

    with st.expander("📝 Dataset Overview & Key Observations"):
        st.dataframe(df.head())
        st.markdown("""
        * **Imbalance:** The dataset is highly imbalanced (approx 5% stroke cases).
        * **Age Factor:** Older age groups show a significantly higher density of stroke occurrences.
        * **Glucose Levels:** Right-skewed distribution; high glucose correlates with stroke risk.
        * **Missing Values:** BMI contained missing values which were imputed with the median.
        """)

    st.markdown("---")
    
    # --- Visualization Section ---
    col1, col2 = st.columns(2)
    
    # 1. Univariate Analysis
    with col1:
        st.subheader("1. Univariate Analysis")
        st.info("Analyze the distribution of a single variable.")
        
        univ_var = st.selectbox("Select Variable", df.columns, index=1)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if pd.api.types.is_numeric_dtype(df[univ_var]) and len(df[univ_var].unique()) > 10:
            sns.histplot(df[univ_var], kde=True, color='teal', ax=ax)
            ax.set_title(f"Distribution of {univ_var}")
        else:
            sns.countplot(x=univ_var, data=df, palette='viridis', ax=ax)
            ax.set_title(f"Count of {univ_var}")
            if len(df[univ_var].unique()) > 5:
                plt.xticks(rotation=45)
                
        st.pyplot(fig)

    # 2. Bivariate Analysis
    with col2:
        st.subheader("2. Bivariate Analysis")
        st.info("Compare two variables to find relationships.")
        
        biv_var_1 = st.selectbox("Select X-Axis", df.columns, index=0)
        biv_var_2 = st.selectbox("Select Y-Axis (or Hue)", df.columns, index=11) # Default to 'stroke'
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        
        # Logic for Plot Type
        is_num_1 = pd.api.types.is_numeric_dtype(df[biv_var_1]) and len(df[biv_var_1].unique()) > 5
        is_num_2 = pd.api.types.is_numeric_dtype(df[biv_var_2]) and len(df[biv_var_2].unique()) > 5
        
        try:
            if is_num_1 and is_num_2:
                # Scatter for Num vs Num
                sns.scatterplot(data=df, x=biv_var_1, y=biv_var_2, hue='stroke', palette='coolwarm', alpha=0.6, ax=ax2)
                ax2.set_title("Scatter Plot (Colored by Stroke)")
            
            elif is_num_1 and not is_num_2:
                # KDE or Box for Num vs Cat
                sns.boxplot(data=df, x=biv_var_2, y=biv_var_1, palette='pastel', ax=ax2)
                ax2.set_title(f"{biv_var_1} Distribution by {biv_var_2}")
                
            elif not is_num_1 and is_num_2:
                # Box for Cat vs Num
                sns.boxplot(data=df, x=biv_var_1, y=biv_var_2, palette='pastel', ax=ax2)
                ax2.set_title(f"{biv_var_2} Distribution by {biv_var_1}")
                
            else:
                # Stacked Bar for Cat vs Cat
                cross_tab = pd.crosstab(df[biv_var_1], df[biv_var_2], normalize='index')
                cross_tab.plot(kind='bar', stacked=True, colormap='viridis', ax=ax2)
                ax2.set_ylabel('Proportion')
                ax2.set_title(f"Relationship: {biv_var_1} vs {biv_var_2}")
                
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Could not plot this combination: {e}")