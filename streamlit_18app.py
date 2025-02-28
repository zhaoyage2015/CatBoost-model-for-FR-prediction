import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model with error handling
try:
    model = joblib.load('CatBoost_model_02-27.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

# 界面布局优化
st.markdown("<h6 style='text-align: center; color: black;'>FR Prediction</h6>", unsafe_allow_html=True)

# 特征顺序验证
REQUIRED_FEATURES = [
    'Pulmonary infection', 'NLR', 'ASPECTS', 'Hypertension', 'Serum glucose',
    'Hemorrhage transformation_1', 'Initial NIHSS', 'Neutrophils', 'CRP',
    'Baseline DBP', 'MLS', 'Age'
]

# 用户输入控件（保持严格顺序）
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

# 验证特征数量
if len(inputs) != len(REQUIRED_FEATURES):
    st.error(f"特征数量不匹配，预期{len(REQUIRED_FEATURES)}个，实际{len(inputs)}个")
    st.stop()

# 构建特征DataFrame
try:
    features = pd.DataFrame([inputs], columns=REQUIRED_FEATURES)
except Exception as e:
    st.error(f"特征矩阵构建失败: {str(e)}")
    st.stop()

# 预测逻辑
if st.button("Predict"):
    try:
        # 执行预测
        predicted = model.predict_proba(features)[0]
        prob_fr = round(predicted[1]*100, 2)
        
        # 显示结果
        st.markdown(f"""
        ### 预测结果
        - **无效再通概率**: {prob_fr}%
        - **有效再通概率**: {round(predicted[0]*100, 2)}%
        """)
        
        # SHAP解释
        with st.spinner("生成解释..."):
            try:
                # 创建内存缓冲区避免文件操作
                buf = BytesIO()
                
                # 创建SHAP解释器
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(features)
                
                # 生成可视化
                plt.figure(figsize=(12, 6))
                shap.plots.waterfall(shap_values[0], max_display=12, show=False)
                plt.tight_layout()
                
                # 保存到内存
                plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                plt.close()
                
                # 显示图像
                buf.seek(0)
                st.image(buf, caption="SHAP解释 (Waterfall Plot)")
                
            except Exception as e:
                st.error(f"SHAP解释生成失败: {str(e)}")
                logger.exception("SHAP error")
                
    except Exception as e:
        st.error(f"预测失败: {str(e)}")
        logger.exception("Prediction error")
