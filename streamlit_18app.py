import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('CatBoost_model_11.pkl')
st.markdown("<h6 style='text-align: center; color: black;'>FR Prediction</h6>", unsafe_allow_html=True)
st.markdown("""
    <style>
    /* 覆盖 Streamlit 默认控件的上下间距 */
    div[data-testid="stVerticalBlock"] {
        gap: 0rem !important; /* 去除垂直块之间的间距 */
    }
    div.stNumberInput, div.stSelectbox, div.stTextInput, div.stSlider {
        margin-bottom: 0rem !important; /* 强制去掉控件底部间距 */
        margin-top: 0rem !important;    /* 强制去掉控件顶部间距 */
    }
    </style>
    """, unsafe_allow_html=True) 
# 定义特征名称
feature_name_mapping = {
    'NLR': 'NLR',
    'Pulmonary infection': 'Pulmonary infection',
    'ASPECTS': 'ASPECTS',
    'Serum glucose': 'Serum glucose',
    'Hypertension': 'Hypertension',
    'Hemorrhagic transformation_1': 'Symptomatic HT',  # 替换名称
    'Initial NIHSS': 'Initial NIHSS',
    'Baseline DBP': 'Baseline DBP',
    'Neutrophils': 'Neutrophils',
    'Age': 'Age',
    'MLS': 'MLS',
    'AGR': 'AGR'
}

# 输入控件
NLR = st.number_input(f"{feature_name_mapping['NLR']}:", min_value=1.00, max_value=100.00, value=10.00)
PI = st.selectbox(f"{feature_name_mapping['Pulmonary infection']} (0=NO, 1=Yes):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'Yes (1)')
ASPECTS = st.number_input(f"{feature_name_mapping['ASPECTS']}:", min_value=5, max_value=10, value=8)
HTN = st.selectbox(f"{feature_name_mapping['Hypertension']} (0=NO, 1=Yes):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'Yes (1)')
GLU = st.number_input(f"{feature_name_mapping['Serum glucose']}:", min_value=2.2, max_value=32.0, value=8.0)
Symptomatic_HT = st.selectbox(f"{feature_name_mapping['Hemorrhagic transformation_1']} (0=NO, 1=Yes):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'Yes (1)')
NIHSS = st.number_input(f"{feature_name_mapping['Initial NIHSS']}:", min_value=3, max_value=42, value=16)
MLS = st.number_input(f"{feature_name_mapping['MLS']}:", min_value=0.00, max_value=30.00, value=2.88)
DBP = st.number_input(f"{feature_name_mapping['Baseline DBP']}:", min_value=40, max_value=160, value=85)
NE = st.number_input(f"{feature_name_mapping['Neutrophils']}:", min_value=1.50, max_value=30.00, value=8.00)
CRP = st.number_input(f"{feature_name_mapping['AGR']}:", min_value=0.10, max_value=200.00, value=12.50)
Age = st.number_input(f"{feature_name_mapping['Age']}:", min_value=18, max_value=100, value=66)

# 特征值列表
feature_values = [NLR, PI, ASPECTS, GLU, HTN, Symptomatic_HT, NIHSS, MLS, DBP, NE, CRP, Age]

# 转换为模型特征
features = pd.DataFrame([feature_values], columns=list(feature_name_mapping.keys()))


features = features[[name for name in feature_name_mapping.keys()]]

# 特征值列表

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
plt.figure(figsize=(20, 10))
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    features.iloc[0, :],  # 使用 DataFrame 的单行，确保顺序对应
    feature_names=features.columns,  # 使用原始特征名与顺序一致
    matplotlib=True,
    show=False
)

# 获取当前图形中的文本元素
ax = plt.gca()
texts = [t for t in ax.texts]  # 提取所有标签文本

# 分别调整 Hypertension 和 Pulmonary infection 的位置
for text in texts:
    if "Hypertension" in text.get_text():
        current_pos = text.get_position()
        text.set_position((current_pos[0] - 0.5, current_pos[1]))  # Hypertension 左移 6mm
    if "Pulmonary infection" in text.get_text():
        current_pos = text.get_position()
        text.set_position((current_pos[0] - 0.4, current_pos[1]))  # Pulmonary infection 左移 6mm

# 保存高分辨率图片
plt.savefig("shap_force_plot_optimized_final.png", bbox_inches='tight', dpi=600)  # 进一步提高分辨率
st.image("shap_force_plot_optimized_final.png", caption="SHAP Force Plot (Further Optimized)")

# 添加 SHAP Summary Plot 作为替代选项
st.subheader("SHAP Summary Plot")
plt.figure(figsize=(10, 6))  # 设置适合论文使用的尺寸
shap.summary_plot(shap_values, features, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=600)  # 保存为高分辨率图片
st.image("shap_summary_plot.png", caption="SHAP Summary Plot")
