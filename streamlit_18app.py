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
        - **Effective Reperfusion Probability**: {round(predicted[0]*100, 2)}%
        """)
        
        # SHAP explanation
        with st.spinner("Generating explanation..."):
            try:
                # Clear previous figure if any
                plt.clf()  # Clear previous figure
                
                # 1. Adjusting plot settings for better clarity and sharpness
                plt.rcParams.update({
                    'font.family': 'Helvetica',  # Use a clean font for publication (Helvetica is widely used)
                    'font.size': 14,             # Increase font size for better readability
                    'lines.linewidth': 2,        # Increase line width for clearer lines
                    'axes.titlepad': 25,         # Increase title padding to avoid crowding
                    'savefig.facecolor': 'white' # White background for the plot to avoid distractions
                })

                # 2. Create SHAP explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(features)
                
                # 3. Handle scalar output (if shap_values is scalar, use it directly)
                if isinstance(shap_values, list):
                    shap_values_class_1 = shap_values[1]  # SHAP values for class 1 (if binary classification)
                    base_value = explainer.expected_value[1]  # Base value for class 1
                else:
                    shap_values_class_1 = shap_values  # For single class output (regression or single-class classification)
                    base_value = explainer.expected_value  # Use the scalar base value
                
                # 4. Generate SHAP Force Plot with higher resolution
                fig = plt.figure(figsize=(18, 10), dpi=600)  # Higher DPI for better sharpness
                ax = fig.add_subplot(111)
                
                force_plot = shap.force_plot(
                    base_value=base_value,
                    shap_values=shap_values_class_1,
                    features=features.iloc[0, :],
                    feature_names=REQUIRED_FEATURES,
                    matplotlib=True,
                    show=False,
                    text_rotation=45,  # Rotate text more for better readability
                    contribution_threshold=0.05  # Filter very small contributions
                )

                # 5. Enhance rendering parameters for better readability
                plt.tight_layout(pad=5.0)  # Add more padding around the plot
                ax.spines['top'].set_visible(False)  # Remove top border for cleaner look
                ax.spines['right'].set_visible(False)  # Remove right border

                # 6. Save high-quality plot to in-memory buffer
                buf = BytesIO()
                plt.savefig(buf, 
                           format='png',
                           dpi=600,  # Increased DPI for sharper image
                           bbox_inches='tight',
                           pad_inches=0.5,
                           transparent=False,
                           facecolor='white')  # Set a white background
                plt.close()

                # Display image
                buf.seek(0)
                st.image(buf, caption="High-Resolution SHAP Force Plot", use_container_width=True)
                
            except Exception as e:
                st.error(f"SHAP explanation generation failed: {str(e)}")
                logger.exception("SHAP error")
                
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        logger.exception("Prediction error")
