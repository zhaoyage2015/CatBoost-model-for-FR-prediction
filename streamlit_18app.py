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

# Model loading with error handling
try:
    model = joblib.load('CatBoost_model_02-27.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

# Interface configuration
st.markdown("<h4 style='text-align: center; margin-bottom: 30px;'>Futile Reperfusion Risk Assessment</h4>", 
            unsafe_allow_html=True)

# Feature validation
REQUIRED_FEATURES = [
    'Pulmonary infection', 'NLR', 'ASPECTS', 'Hypertension', 'Serum glucose',
    'Hemorrhage transformation_1', 'Initial NIHSS', 'Neutrophils', 'CRP',
    'Baseline DBP', 'MLS', 'Age'
]

# Clinical parameter inputs
inputs = []
with st.container():
    cols = st.columns(3)
    
    # Column 1
    with cols[0]:
        inputs.append(st.selectbox("Pulmonary infection [0=No, 1=Yes]", 
                                 options=[0, 1]))
        inputs.append(st.number_input("Neutrophil-to-Lymphocyte Ratio (NLR)", 
                                   min_value=1.0, max_value=100.0, value=10.0, step=0.1))
        inputs.append(st.number_input("ASPECTS Score", 
                                   min_value=5, max_value=10, value=7))
        inputs.append(st.selectbox("Hypertension History [0=No, 1=Yes]", 
                                 options=[0, 1]))
    
    # Column 2
    with cols[1]:
        inputs.append(st.number_input("Serum Glucose (mmol/L)", 
                                   min_value=2.2, max_value=32.0, value=8.0, step=0.1))
        inputs.append(st.selectbox("Symptomatic Hemorrhagic Transformation [0=No, 1=Yes]", 
                                 options=[0, 1]))
        inputs.append(st.number_input("NIHSS Score at Admission", 
                                   min_value=3, max_value=42, value=17))
        inputs.append(st.number_input("Neutrophil Count (×10⁹/L)", 
                                   min_value=1.5, max_value=30.0, value=9.0, step=0.1))
    
    # Column 3
    with cols[2]:
        inputs.append(st.number_input("C-reactive Protein (mg/L)", 
                                   min_value=0.1, max_value=200.0, value=12.5, step=0.1))
        inputs.append(st.number_input("Diastolic Blood Pressure (mmHg)", 
                                   min_value=40, max_value=160, value=85))
        inputs.append(st.number_input("Midline Shift (mm)", 
                                   min_value=0.0, max_value=30.0, value=2.88, step=0.01))
        inputs.append(st.number_input("Age (Years)", 
                                   min_value=18, max_value=100, value=66))

# Feature validation
if len(inputs) != len(REQUIRED_FEATURES):
    st.error(f"Feature mismatch: Expected {len(REQUIRED_FEATURES)}, got {len(inputs)}")
    st.stop()

# Create feature matrix
try:
    features = pd.DataFrame([inputs], columns=REQUIRED_FEATURES)
except Exception as e:
    st.error(f"Feature matrix construction failed: {str(e)}")
    st.stop()

# Prediction logic
if st.button("Calculate FR Risk"):
    try:
        # Generate predictions
        pred_proba = model.predict_proba(features)[0]
        fr_prob = round(pred_proba[1]*100, 2)
        er_prob = round(pred_proba[0]*100, 2)
        
        # Display results
        st.markdown(f"""
        ### Prediction Results
        - **FR Probability**: {fr_prob}%
        - **Effective Reperfusion Probability**: {er_prob}%
        """)
        
        # SHAP interpretation
        with st.spinner("Generating explanation..."):
            try:
                explainer = shap.TreeExplainer(model)
                expected_value = explainer.expected_value[1]  # FR class
                shap_values = explainer.shap_values(features)[1][0]  # FR class values
                
                # Create force plot
                plt.figure(figsize=(12, 4))
                shap.force_plot(
                    base_value=expected_value,
                    shap_values=shap_values,
                    features=features.iloc[0],
                    feature_names=REQUIRED_FEATURES,
                    matplotlib=True,
                    show=False,
                    text_rotation=15
                )
                
                # Format plot
                plt.title("SHAP Force Plot for FR Prediction", pad=20)
                plt.tight_layout()
                
                # Display in Streamlit
                st.pyplot(plt.gcf())
                plt.close()
                
                # Add interpretation guide
                st.caption("""
                **Force Plot Interpretation**:
                - Red features increase FR risk
                - Blue features decrease FR risk
                - Base value represents population-average risk ({:.1f}%)
                """.format(100/(1 + np.exp(-expected_value))))
                
            except Exception as e:
                st.error(f"SHAP interpretation failed: {str(e)}")
                logger.exception("SHAP error")
                
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        logger.exception("Calculation error")
