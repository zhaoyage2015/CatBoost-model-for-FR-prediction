import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load model
model = joblib.load('CatBoost_model_02-27.pkl')
st.markdown("<h6 style='text-align: center; color: black;'>FR Prediction</h6>", unsafe_allow_html=True)

st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"] {
        gap: 0rem !important;
    }
    div.stNumberInput, div.stSelectbox, div.stTextInput, div.stSlider {
        margin-bottom: 0rem !important;
        margin-top: 0rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Feature name mapping
feature_name_mapping = {
    'Pulmonary infection': 'Pulmonary infection',
    'NLR': 'NLR',
    'ASPECTS': 'ASPECTS',
    'Hypertension': 'Hypertension',
    'Serum glucose': 'Serum glucose',
    'Hemorrhage transformation_1': 'Symptomatic HT',
    'Initial NIHSS': 'Initial NIHSS',
    'Neutrophils': 'Neutrophils',
    'CRP': 'CRP',
    'Baseline DBP': 'Baseline DBP',
    'MLS': 'MLS',
    'Age': 'Age'
}

# User input controls
PI = st.selectbox(f"{feature_name_mapping['Pulmonary infection']} (0=NO, 1=Yes):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'Yes (1)')
NLR = st.number_input(f"{feature_name_mapping['NLR']}:", min_value=1.00, max_value=100.00, value=10.00)
ASPECTS = st.number_input(f"{feature_name_mapping['ASPECTS']}:", min_value=5, max_value=10, value=8)
HTN = st.selectbox(f"{feature_name_mapping['Hypertension']} (0=NO, 1=Yes):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'Yes (1)')
GLU = st.number_input(f"{feature_name_mapping['Serum glucose']}:", min_value=2.2, max_value=32.0, value=8.0)
Symptomatic_HT = st.selectbox(f"{feature_name_mapping['Hemorrhage transformation_1']} (0=NO, 1=Yes):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'Yes (1)')
NIHSS = st.number_input(f"{feature_name_mapping['Initial NIHSS']}:", min_value=3, max_value=42, value=17)
NE = st.number_input(f"{feature_name_mapping['Neutrophils']}:", min_value=1.50, max_value=30.00, value=8.00)
CRP = st.number_input(f"{feature_name_mapping['CRP']}:", min_value=0.10, max_value=200.00, value=12.50)
DBP = st.number_input(f"{feature_name_mapping['Baseline DBP']}:", min_value=40, max_value=160, value=85)
MLS = st.number_input(f"{feature_name_mapping['MLS']}:", min_value=0.00, max_value=30.00, value=2.88)
Age = st.number_input(f"{feature_name_mapping['Age']}:", min_value=18, max_value=100, value=66)

# Feature values (ensure correct order!)
feature_values = [PI, NLR, ASPECTS, HTN, GLU, Symptomatic_HT, NIHSS, NE, CRP, DBP, MLS, Age]

# Corrected feature names for the model
model_features = [
    'Pulmonary infection', 'NLR', 'ASPECTS', 'Hypertension', 'Serum glucose',
    'Hemorrhage transformation_1', 'Initial NIHSS', 'Neutrophils', 'CRP',
    'Baseline DBP', 'MLS', 'Age'
]

# Prepare features DataFrame
features = pd.DataFrame([feature_values], columns=model_features)

# Prediction button
if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = round(predicted_proba[predicted_class] * 100, 2)

    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {[round(p * 100, 2) for p in predicted_proba]}")

    result = f"According to feature values, predicted possibility of FR is: {probability}%" \
        if predicted_class == 1 else \
        f"According to feature values, predicted possibility of FR is: {100 - probability}%"
    st.write(result)

# SHAP Force Plot generation
# SHAP Force Plot generation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features)

plt.figure(figsize=(22, 8))  # 加宽画布水平空间
ax = plt.gca()

# 生成基础force plot
shap.force_plot(
    base_value=explainer.expected_value,
    shap_values=shap_values[0],
    features=features.iloc[0, :],
    feature_names=model_features,
    matplotlib=True,
    show=False,
    text_rotation=15,  # 添加标签旋转
    plot_cmap="coolwarm"  # 调整颜色增强可读性
)

# 手动调整标签布局
def adjust_shap_labels(ax, x_offset=0.05):
    """调整SHAP标签位置防止重叠"""
    texts = [t for t in ax.texts if t.get_text() in model_features]
    
    # 计算初始位置
    y_positions = np.linspace(0.1, 0.9, len(texts))
    
    for i, text in enumerate(texts):
        # 设置垂直对齐和旋转
        text.set_rotation(12)  # 轻微旋转
        text.set_ha('right')   # 水平对齐方式
        text.set_va('center')  # 垂直对齐
        
        # 动态水平偏移
        current_x = text.get_position()[0]
        text.set_position((current_x - (i%3)*x_offset, y_positions[i]))

adjust_shap_labels(ax, x_offset=0.08)

# 优化保存参数
plt.savefig("shap_force_plot_final.png", 
           bbox_inches='tight', 
           dpi=600, 
           pad_inches=0.2)  # 增加边距
st.image("shap_force_plot_final.png", caption="SHAP Force Plot (Optimized Layout)")
