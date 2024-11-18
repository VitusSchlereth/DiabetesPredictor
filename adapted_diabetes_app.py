import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
from PIL import Image
from streamlit_lottie import st_lottie
import requests

# Function to load Lottie animation
def load_lottie_url(url:str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load the diabetes dataset
@st.cache_data()
def load_data():
    df = pd.read_csv('./data/diabetes.csv')
    return df

# Set up page layout
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Overview", "Model Prediction"])

# Display a Lottie animation in the main area
lottie_animation = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_8IuO76.json")
st_lottie(lottie_animation, speed=1, width=700, height=400, key="animation")

# Home page
if page == "Home":
    st.title("Diabetes Prediction System")
    st.markdown("""
    This app predicts whether a person has diabetes based on various health parameters.
    The machine learning model uses historical data of diabetes patients to make predictions.
    """)

# Data Overview
elif page == "Data Overview":
    st.title("Dataset Overview")
    df = load_data()
    st.write(df.head())  # Display first few rows of data

    # Interactive Plot with Plotly
    st.subheader("Interactive Data Visualization")
    fig = px.scatter(df, x="Glucose", y="Insulin", color="Outcome", title="Glucose vs Insulin")
    st.plotly_chart(fig)

    # Add some more visuals
    st.subheader("Correlation Heatmap")
    corr_matrix = df.corr()
    fig2 = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap")
    st.plotly_chart(fig2)

# Model Prediction
elif page == "Model Prediction":
    st.title("Diabetes Prediction")
    st.markdown("""
    Predict whether a person has diabetes based on the following features:
    """)
    
    # Inputs from user
    age = st.number_input("Age", min_value=1, max_value=120)
    glucose = st.number_input("Glucose Level", min_value=1, max_value=200)
    insulin = st.number_input("Insulin Level", min_value=1, max_value=200)
    
    # More inputs here...

    # Prediction button
    if st.button("Predict"):
        # Placeholder for model prediction code (you can replace it with your model)
        st.success(f"Prediction result: {'Diabetic' if glucose > 120 else 'Non-Diabetic'}")

# Footer with some information
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This is a machine learning app that predicts diabetes based on various health parameters.
It was built with Streamlit, Python, and machine learning algorithms.
""")
