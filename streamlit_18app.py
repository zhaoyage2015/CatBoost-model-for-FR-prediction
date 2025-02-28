import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model with error handling
try:
    model = joblib.load('CatBoost_model_02-27.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

# Interface layout optimization
st.markdown("<h6 style='text-align: center; color: black;'>FR Prediction</h6>", unsafe_allow_html=True)

# Feature order validation
REQUIRED_FEATURES = [
    'Pulmonary infection', 'NLR', 'ASPECTS', 'Hypertension', 'Serum glucose',
    'Hemorrhage transformation_1', 'Initial NIHSS', 'Neutrophils', 'CRP',
    'Baseline DBP', 'MLS', 'Age'
]

# User input controls (maintain strict order)
inputs = []
with st.container():
    cols = st.columns(3)
    
    # Column 1
    with cols[0]:
        inputs.append(st.selectbox("Pulmonary infection (0=NO, 1=Yes)", 
                                 options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'Yes (1)'))
        inputs.append(st.number_input("NLR", min_value=1.0, max_value=100.0, value=10.0, step=0.1))
        inputs.append(st.number_input("ASPECTS", min_value=5, max_value=10, value=7))
        inputs.append(st.selectbox("Hypertension (0=NO, 1=Yes)", 
                                 options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'Yes (1)'))
    
    # Column 2
    with cols[1]:
        inputs.append(st.number_input("Serum glucose (mmol/L)", min_value=2.2, max_value=32.0, value=8.0, step=0.1))
        inputs.append(st.selectbox("Symptomatic HT (0=NO, 1=Yes)", 
                                 options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'Yes (1)'))
        inputs.append(st.number_input("Initial NIHSS", min_value=3, max_value=42, value=17))
        inputs.append(st.number_input("Neutrophils (×10⁹/L)", min_value=1.5, max_value=30.0, value=9.0, step=0.1))
    
    # Column 3
    with cols[2]:
        inputs.append(st.number_input("CRP (mg/L)", min_value=0.1, max_value=200.0, value=12.5, step=0.1))
        inputs.append(st.number_input("Baseline DBP (mmHg)", min_value=40, max_value=160, value=85))
        inputs.append(st.number_input("MLS (mm)", min_value=0.0, max_value=30.0, value=2.88, step=0.01))
        inputs.append(st.number_input("Age", min_value=18, max_value=100, value=66))

# Validate feature count
if len(inputs) != len(REQUIRED_FEATURES):
    st.error(f"Feature count mismatch, expected {len(REQUIRED_FEATURES)}, but got {len(inputs)}")
    st.stop()

# Construct feature DataFrame
try:
    features = pd.DataFrame([inputs], columns=REQUIRED_FEATURES)
except Exception as e:
    st.error(f"Feature matrix construction failed: {str(e)}")
    st.stop()

# Prediction logic
if st.button("Predict"):
    try:
        # Perform prediction
        predicted = model.predict_proba(features)[0]
        prob_fr = round(predicted[1]*100, 2)
        
        # Display results
        st.markdown(f"""
        ### Prediction Results
        - **FR Probability**: {prob_fr}%
        - **Successful Reperfusion Probability**: {round(predicted[0]*100, 2)}%
        """)
        
        # SHAP explanation
        with st.spinner("Generating explanation..."):
            try:
                # Create in-memory buffer to avoid file operations
                buf = BytesIO()
                
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(features)
                
                # Handle scalar output (if shap_values is scalar, use it directly)
                if isinstance(shap_values, list):
                    shap_values_class_1 = shap_values[1]  # SHAP values for class 1 (if binary classification)
                    base_value = explainer.expected_value[1]  # Base value for class 1
                else:
                    shap_values_class_1 = shap_values  # For single class output (regression or single-class classification)
                    base_value = explainer.expected_value  # Use the scalar base value
                
                # Generate SHAP Force Plot
                plt.figure(figsize=(20, 10))
                shap.force_plot(
                    base_value=base_value,  # Use the base value for class 1 (for binary classification) or scalar base value
                    shap_values=shap_values_class_1,  # Use SHAP values for class 1
                    features=features.iloc[0, :],  # Ensure correct mapping of features
                    feature_names=REQUIRED_FEATURES,
                    matplotlib=True,
                    show=False
                )
                plt.tight_layout()
                
                # Save to in-memory
                plt.savefig(buf, format="png", dpi=1200, bbox_inches="tight")
                plt.close()
                
                # Display image
                buf.seek(0)
                st.image(buf, caption="SHAP Force Plot")
                
            except Exception as e:
                st.error(f"SHAP explanation generation failed: {str(e)}")
                logger.exception("SHAP error")
                
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        logger.exception("Prediction error")
