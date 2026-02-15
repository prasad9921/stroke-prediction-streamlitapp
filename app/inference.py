import streamlit as st
import pandas as pd
import numpy as np
import joblib

def show(preprocessor, metadata):
    st.header("🩺 Live Prediction Lab")
    st.markdown("Enter patient details below to predict stroke risk.")
    
    # --- Input Form ---
    with st.form("inference_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
            age = st.number_input("Age", min_value=1, max_value=100, value=30)
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
        with col2:
            heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            ever_married = st.selectbox("Ever Married", ['Yes', 'No'])
            work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
            
        with col3:
            residence = st.selectbox("Residence Type", ['Urban', 'Rural'])
            avg_glucose = st.number_input("Avg Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
            smoking = st.selectbox("Smoking Status", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

        st.markdown("---")
        model_choice = st.selectbox("Choose AI Model for Diagnosis", list(metadata['models'].keys()))
        
        submit_btn = st.form_submit_button("Analyze Patient Data")

    # --- Prediction Logic ---
    if submit_btn:
        # 1. Prepare Raw DataFrame (Must match the columns used in training BEFORE transform)
        # Note: We dropped 'id' and 'stroke' in training. 
        # Crucial: Training used 'avg_glucose_log' in numerical features.
        
        # We need to manually calculate log of glucose because our preprocessor expects 'avg_glucose_log'
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'Residence_type': [residence],
            'avg_glucose_log': [np.log(avg_glucose)], # Apply Log Transform here
            'bmi': [bmi],
            'smoking_status': [smoking]
        })
        
        # 2. Preprocess
        try:
            X_processed = preprocessor.transform(input_data)
        except Exception as e:
            st.error(f"Preprocessing Error: {e}")
            st.stop()
            
        # 3. Load Model
        model_info = metadata['models'][model_choice]
        model_path = model_info['model_path']
        
        try:
            model = joblib.load(model_path)
        except:
            st.error(f"Model file {model_path} not found.")
            st.stop()
            
        # 4. Predict
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0][1]
        
        # 5. Display Results
        st.markdown("### 🔍 Diagnostic Results")
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            if prediction == 1:
                st.error("⚠️ HIGH RISK DETECTED")
                st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=100)
            else:
                st.success("✅ LOW RISK DETECTED")
                st.image("https://cdn-icons-png.flaticon.com/512/272/272449.png", width=100)
                
        with c2:
            st.metric("Stroke Probability", f"{round(probability * 100, 2)}%")
            st.progress(float(probability))
            
            if probability > 0.5:
                st.warning("The model suggests a high likelihood of stroke based on the provided parameters. Please consult a medical professional.")
            else:
                st.info("The model suggests a low likelihood. However, maintaining a healthy lifestyle is always recommended.")