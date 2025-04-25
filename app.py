import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Page Config
st.set_page_config(page_title="🧠 Mental Health Predictor", layout="centered")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("smmh_final_preprocessed.csv")

df = load_data()
features = [col for col in df.columns if col != '18. How often do you feel depressed or down?']
target = '18. How often do you feel depressed or down?'

# Load model
model = joblib.load("best_model.pkl")

# App Header
st.title("🧠 Social Media & Mental Health Impact Analysis")
st.markdown("Welcome to the Mental Health Predictor based on social media and lifestyle behaviors.")

# Sidebar Navigation
page = st.sidebar.radio("🔍 Navigation", ["📊 EDA", "🔎 Features", "🧠 Predict Depression"])

# 1. EDA
if page == "📊 EDA":
    st.header("📊 Exploratory Data Analysis")
    st.dataframe(df.head(), use_container_width=True)

    if st.checkbox("📈 Show Summary Statistics"):
        st.write(df.describe())

    if st.checkbox("🌡️ Show Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# 2. Feature Overview
elif page == "🔎 Features":
    st.header("📌 Model Features Overview")
    st.write(f"✅ Total Features used for training: `{len(features)}`")
    st.markdown("Here are the feature columns (excluding target variable):")
    st.code(features)

# 3. Prediction
elif page == "🧠 Predict Depression":
    st.header("🔮 Predict Risk of Depression")

    st.markdown("Fill the form below with your details 👇")

    user_input = {}
    for feat in features:
        if df[feat].nunique() == 2:
            user_input[feat] = st.radio(f"{feat}", [0, 1], index=0, help="0 = No, 1 = Yes")
        else:
            user_input[feat] = st.slider(f"{feat}", float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()))

    input_df = pd.DataFrame([user_input])

    if st.button("🔍 Predict Now"):
        prediction = model.predict(input_df)[0]
        if prediction >= 3:
            st.error("⚠️ High Risk of Depression. Please consider seeking support.")
        elif prediction == 2:
            st.warning("🟠 Moderate Risk of Depression.")
        else:
            st.success("✅ Low Risk of Depression. Keep taking care of your mental health!")

        st.markdown("📌 **Note:** This prediction is based on statistical modeling. It is not a substitute for professional advice.")

# Footer
st.markdown("---")
st.caption("© 2025 - Created with by Ankita")
