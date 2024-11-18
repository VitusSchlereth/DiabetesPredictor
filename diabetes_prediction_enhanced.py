
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image

# Function to load the diabetes dataset
def load_data():
    df = pd.read_csv('./data/diabetes.csv')
    return df

# Load the pre-trained model
def load_model():
    with open('./data/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Load the pre-fitted scaler
def load_scaler():
    with open('./data/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler

# Function to make prediction
def make_prediction(model, scaler, inputs):
    data = {
        "Pregnancies": inputs[0],
        "Glucose": inputs[1],
        "BloodPressure": inputs[2],
        "SkinThickness": inputs[3],
        "Insulin": inputs[4],
        "BMI": inputs[5],
        "DiabetesPedigreeFunction": inputs[6],
        "Age": inputs[7]
    }
    
    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df)
    df_scaled_np = np.array(df_scaled)
    
    prediction = model.predict(df_scaled_np)
    
    return prediction

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Sample user inputs
pregnancies = 1
glucose = 100
blood_pressure = 80
skin_thickness = 20
insulin = 50
bmi = 25.0
diabetes_pedigree_function = 0.5
age = 30

# Load the model and scaler
model = load_model()
scaler = load_scaler()
inputs = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]

# Make prediction
prediction = make_prediction(model, scaler, inputs)

# Output prediction result
if prediction == 1:
    print("The model predicts: Diabetic")
else:
    print("The model predicts: Not Diabetic")

# Model evaluation (sample)
y_true = [0, 1, 1, 0, 1, 0, 1]  # Sample actual values
y_pred = [0, 1, 0, 0, 1, 0, 1]  # Sample predicted values

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred)
