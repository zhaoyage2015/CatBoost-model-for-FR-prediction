import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('CatBoost_model_11.pkl')
st.title("FR Prediction")

# 定义特征名称
feature_names = ['NLR', 'Pulmonary infection', 'ASPECTS', 'Serum glucose', 'Hypertension',
                 'Hemorrhagic transformation_1', 'Initial NIHSS', 'Baseline DBP',
                 'Neutrophils', 'Age', 'MLS', 'AGR']

# 输入控件
NLR = st.number_input("NLR:", min_value=1.00, max_value=100.00, value=10.00)
PI = st.selectbox("Pulmonary infection (0=NO, 1=Yes):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'Yes (1)')
ASPECTS = st.number_input("ASPECTS:", min_value=5, max_value=10, value=8)
HTN = st.selectbox("Hypertension (0=NO, 1=Yes):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'Yes (1)')
GLU = st.number_input("Serum glucose:", min_value=2.2, max_value=32.0, value=8.0)
SHT = st.selectbox("Hemorrhagic transformation_1 (0=NO, 1=Yes):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'Yes (1)')
NIHSS = st.number_input("Initial NIHSS:", min_value=3, max_value=42, value=16)
MLS = st.number_input("MLS:", min_value=0.00, max_value=30.00, value=2.88)
DBP = st.number_input("Baseline DBP:", min_value=40, max_value=160, value=85)
NE = st.number_input("Neutrophils:", min_value=1.50, max_value=30.00, value=8.00)
CRP = st.number_input("CRP:", min_value=0.10, max_value=200.00, value=12.50)
Age = st.number_input("Age:", min_value=18, max_value=100, value=66)

# 特征值列表
feature_values = [NLR, PI, ASPECTS, GLU, HTN, SHT, NIHSS, DBP, NE, Age, MLS, CRP]
features = pd.DataFrame([feature_values], columns=feature_names)

# 预测按钮
if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = round(predicted_proba[predicted_class] * 100, 2)  # 保留两位小数

    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {[round(p * 100, 2) for p in predicted_proba]}")
    
    if predicted_class == 1:
        result = f"According to feature values, predicted possibility of RF is: {probability}%"
    else:
        result = f"According to feature values, predicted possibility of RF is: {100 - probability}%"
    st.write(result)

# SHAP 解释器
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features)

# 增大图形尺寸，解决特征重叠问题
plt.figure(figsize=(16, 8))  # 增大图宽度和高度
shap.force_plot(
    explainer.expected_value, 
    shap_values[0], 
    features.iloc[0, :],  # 使用 DataFrame 的单行，确保特征名称与值匹配
    matplotlib=True, 
    show=False
)

# 保存高分辨率图片
plt.savefig("shap_force_plot_optimized.png", bbox_inches='tight', dpi=300)  # 优化 DPI
st.image("shap_force_plot_optimized.png", caption="SHAP Force Plot (Optimized)")

# 添加 SHAP Summary Plot 作为替代选项
st.subheader("SHAP Summary Plot")
plt.figure(figsize=(10, 6))  # 设置适合论文使用的尺寸
shap.summary_plot(shap_values, features, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=300)  # 保存为高分辨率图片
st.image("shap_summary_plot.png", caption="SHAP Summary Plot")
