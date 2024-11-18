
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from streamlit_lottie import st_lottie
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from PIL import Image
import requests
import matplotlib.pyplot as plt

# Load Lottie animation function
def load_lottie_url(url:str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load the diabetes dataset
@st.cache_data()
def load_data():
    df = pd.read_csv('./data/diabetes.csv')  # Ensure the correct path
    return df

# Page setup
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🍏", layout="wide")

# Sidebar
with st.sidebar:
    st_lottie(load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_dy3evqpm.json"), speed=1, width=250, height=250, key="sidebar_animation")
    st.title("Diabetes Prediction")
    st.write("This app predicts diabetes based on the dataset.")
    st.markdown("[Data Source](https://www.kaggle.com/uciml/pima-indians-diabetes-database)")

# Load data
df = load_data()

# Main title
st.title("Diabetes Prediction Model")
st.write("This is a machine learning model that predicts whether a person has diabetes or not based on various health indicators.")

# Display dataset (if needed)
if st.checkbox("Show Data"):
    st.dataframe(df)

# Data Preprocessing
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training and prediction (Here, we assume a simple logistic regression)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Model evaluation
st.subheader("Model Evaluation")
conf_matrix = confusion_matrix(y_test, y_pred)
st.write("Classification Report")
st.text(classification_report(y_test, y_pred))
st.write("F1 Score: ", f1_score(y_test, y_pred))

# Confusion matrix plot
st.subheader("Confusion Matrix")
fig = px.imshow(conf_matrix, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
st.plotly_chart(fig)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:,1])
roc_auc = auc(fpr, tpr)

st.subheader("ROC Curve")
fig_roc = plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
st.pyplot(fig_roc)

# Correlation matrix
st.subheader("Correlation Matrix")
correlation = df.corr()
fig_corr = plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
st.pyplot(fig_corr)

# Additional visualizations
st.subheader("Data Visualizations")
fig = px.scatter(df, x='Glucose', y='BloodPressure', color='Outcome', title="Glucose vs Blood Pressure")
st.plotly_chart(fig)

# Interactive user input for prediction
st.subheader("Predict the Outcome for a New Patient")
age = st.slider("Age", 18, 100, 25)
glucose = st.slider("Glucose Level", 50, 200, 100)
bp = st.slider("Blood Pressure", 40, 150, 80)
skin_thickness = st.slider("Skin Thickness", 10, 50, 20)
insulin = st.slider("Insulin Level", 10, 500, 100)
bmi = st.slider("BMI", 10, 50, 25)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
pregnancies = st.slider("Pregnancies", 0, 15, 1)

# Create a prediction input array
input_data = np.array([age, glucose, bp, skin_thickness, insulin, bmi, dpf, pregnancies]).reshape(1, -1)
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

if st.button("Predict"):
    if prediction == 1:
        st.success("The model predicts: Diabetes Positive")
    else:
        st.success("The model predicts: Diabetes Negative")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Your Name")
