import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import ast

def show(metadata):
    st.header("🤖 Model Performance & Internals")
    
    model_names = list(metadata['models'].keys())
    
    # Use st.pills for a cleaner, modern selection UI
    selected_model = st.pills("Select a Model to Inspect", model_names, selection_mode="single", default=model_names[-1])
    
    if selected_model:
        model_data = metadata['models'][selected_model]
    
    # 1. Hyperparameters
    with st.expander("⚙️ Model Hyperparameters", expanded=False):
        # Convert Python string dict → actual dict → JSON display
        try:
            params_dict = ast.literal_eval(model_data['parameters'])
            st.json(params_dict, expanded=True)
        except (ValueError, SyntaxError):
            # Fallback if parsing fails
            st.code(model_data['parameters'], language='python')
    
    # 2. Metrics Display
    st.subheader("🏆 Performance Metrics (Test Set)")
    metrics = model_data['final_metrics']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", metrics['Accuracy'])
    col2.metric("F1 Score", metrics['F1'])
    col3.metric("Precision", metrics['Precision'])
    col4.metric("Recall", metrics['Recall'])
    col5.metric("AUC Score", metrics['AUC'])
    
    # 3. Training Curve
    st.subheader("📈 Training Dynamics")
    history = model_data['training_history']
    
    if 'train_acc' in history or 'train_logloss' in history:
        fig, ax = plt.subplots(figsize=(10, 4))
        epochs = history['epoch']
        
        # Handle different metric names (acc vs logloss)
        if 'train_acc' in history:
            ax.plot(epochs, history['train_acc'], label='Training Accuracy', marker='o', color='blue')
            ax.set_ylabel("Accuracy")
        elif 'train_logloss' in history:
             ax.plot(epochs, history['train_logloss'], label='Training LogLoss', marker='o', color='red')
             if 'test_logloss' in history:
                 ax.plot(epochs, history['test_logloss'], label='Test LogLoss', linestyle='--', color='orange')
             ax.set_ylabel("Log Loss")

        ax.set_xlabel("Epochs / Iterations")
        ax.set_title(f"Learning Curve: {selected_model}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("No iterative training history available for this model.")

    # Comparison Table
    st.markdown("---")
    st.subheader("⚔️ All Models Comparison")
    
    comp_data = []
    for name, data in metadata['models'].items():
        row = data['final_metrics']
        row['Model'] = name
        comp_data.append(row)
        
    comp_df = pd.DataFrame(comp_data).set_index('Model')
    st.dataframe(comp_df.style.highlight_max(axis=0, color='lightgreen'))