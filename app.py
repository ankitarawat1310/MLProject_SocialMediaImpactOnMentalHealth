import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("smmh_final_preprocessed.csv")

df = load_data()
features = [col for col in df.columns if col != '18. How often do you feel depressed or down?']
target = '18. How often do you feel depressed or down?'

# Load trained model
model = joblib.load("best_model.pkl")  # Ensure this file exists

# App Title
st.title("📊 Predicting Depression from Social Media and Lifestyle Factors")

# Navigation
option = st.sidebar.radio("Go to", ["EDA", "Feature Overview", "Predict Depression"])

# EDA Section
if option == "EDA":
    st.header("Exploratory Data Analysis")
    st.write("Dataset Preview")
    st.dataframe(df.head())

    if st.checkbox("Summary Statistics"):
        st.write(df.describe())

    if st.checkbox("Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df.corr(), annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# Feature Section
elif option == "Feature Overview":
    st.header("Selected Features for Prediction")
    st.write(f"Total Features: {len(features)}")
    st.write(features)

# Prediction Section
elif option == "Predict Depression":
    st.header("Depression Prediction")

    input_data = {}
    for feat in features:
        if df[feat].nunique() == 2:  # Binary feature
            input_data[feat] = st.selectbox(f"{feat}", [0, 1])
        else:
            input_data[feat] = st.slider(f"{feat}", float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()))

    # Prediction
    input_df = pd.DataFrame([input_data])
    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Depression Level: {prediction}")
