# Diabetes Prediction using Random Forest

## Overview
This project aims to predict the likelihood of diabetes based on several health parameters using a **Random Forest classifier**. The app takes user input for various health metrics and provides a prediction on whether the user is likely to have diabetes or not. The model was trained using a dataset of medical records, with features like glucose levels, BMI, and age playing a key role in the predictions.

The app implements a **Random Forest classifier**, an ensemble learning technique, which improves predictive performance by combining multiple decision trees. This model is evaluated using key performance metrics, such as the confusion matrix, classification report, and ROC curve.

Check out the deployed app here: [Diabetes Predictor App](https://diabetespredictor-vitus.streamlit.app/)

## Dataset Description

The dataset used in this project is based on **diabetes patients** and contains various health-related features. It is a binary classification problem, where the task is to predict if a person has diabetes based on the following variables.

### Dataset Variables:
1. **Pregnancies**: The number of pregnancies the patient has had.
2. **Glucose**: The plasma glucose concentration after two hours in an oral glucose tolerance test.
3. **BloodPressure**: Diastolic blood pressure (mm Hg).
4. **SkinThickness**: Triceps skin fold thickness (mm).
5. **Insulin**: 2-Hour serum insulin (mu U/ml).
6. **BMI (Body Mass Index)**: Weight in kg/(height in m)^2.
7. **DiabetesPedigreeFunction**: A function that represents the genetic relationship between the patient and their family history of diabetes.
8. **Age**: The age of the patient in years.

### Target Variable:
- **Outcome**: A binary variable indicating the presence of diabetes:
  - **0**: No diabetes
  - **1**: Diabetes

## Approach

### Model:
A **Random Forest classifier** is used for this diabetes prediction task. Random Forest is a powerful ensemble learning method that aggregates the predictions of several decision trees to improve the model's accuracy. It is particularly effective for handling complex datasets with multiple features.

### Steps Taken:
1. **Data Preprocessing**:
   - The dataset was cleaned, and missing values were handled as needed.
   - Features were scaled using **StandardScaler** to standardize the data, making it easier for the model to process.
   - The data was split into training and testing sets.

2. **Model Training**:
   - A **Random Forest classifier** was trained using the training set, with a focus on correctly predicting diabetes (or no diabetes).
   - Hyperparameters were not tuned explicitly, but the default Random Forest settings provided reasonable performance.

3. **Model Evaluation**:
   The performance of the trained model was evaluated using several metrics:
   - **Confusion Matrix**: Visualizes the number of true positives, false positives, true negatives, and false negatives.
   - **Classification Report**: Shows precision, recall, and F1-score for both classes (diabetes and no diabetes).
   - **ROC Curve**: Displays the performance of the model by plotting the true positive rate against the false positive rate at various thresholds.

4. **Streamlit Web App**:
   - A **Streamlit** app was developed to allow users to input health metrics and receive a prediction on whether they have diabetes.
   - The app displays relevant evaluation metrics, such as the confusion matrix and ROC curve, as images.

## Requirements

To run this app, you need the following libraries:

- **streamlit**: For building the interactive web app.
- **pandas**: For handling and processing the dataset.
- **numpy**: For numerical computations.
- **scikit-learn**: For building and evaluating the Random Forest classifier.
- **matplotlib**: For generating plots and visualizations.
- **seaborn**: For statistical plots like the confusion matrix and correlation matrix.
- **Pillow**: For image handling and display.
