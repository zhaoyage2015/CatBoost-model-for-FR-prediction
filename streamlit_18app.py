import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('CatBoost_model_11.pkl')
st.title("RF Prediction")

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
feature_values = [NLR, PI, ASPECTS, HTN, GLU, SHT, NIHSS, MLS, DBP, NE, CRP, Age]
features = np.array([feature_values])

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
shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

# 获取所有特征的真实输出值（即完整的 SHAP 值和 expected value）
expected_value = explainer.expected_value
full_shap_values = shap_values[0]

# 选择前8个重要特征的 SHAP 值用于显示
importance_order = np.argsort(-np.abs(full_shap_values))[:8]  # 获取前8个重要特征的索引
top_shap_values = full_shap_values[importance_order]  # 提取前8个特征的SHAP值
top_feature_values = [feature_values[i] for i in importance_order]  # 提取对应的特征值
top_feature_names = [feature_names[i] for i in importance_order]  # 提取对应的特征名称

# 设置字体大小和图像大小
plt.rcParams.update({'font.size': 14})  # 增大字体，确保在图像中更加清晰
plt.figure(figsize=(12, 4))  # 调整图像尺寸以减小内存使用

# 创建 SHAP force plot 并保存
shap.force_plot(
    expected_value, full_shap_values,  # 保留完整的 SHAP 计算，以保持真实输出值
    pd.DataFrame([feature_values], columns=feature_names), 
    matplotlib=True, 
    show=False
)

# 设置坐标轴，使其均匀分布
plt.xlim(-5, 10)  # 设置x轴范围，保证输出图均匀分布
plt.xticks(range(-5, 11, 1))  # 设置x轴的刻度间隔为1，使其更均匀

# 保存图像为高分辨率 PNG 文件
file_name = "shap_force_plot.png"
plt.savefig(file_name, bbox_inches='tight', dpi=600)  # 使用较低 DPI 减小图像大小
st.image(file_name)
