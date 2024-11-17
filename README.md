# DiabetesPredictor
Diabetes Prediction using Random Forest
Overview
This project predicts the likelihood of diabetes based on several health parameters using a Random Forest classifier. The app takes user input for various health metrics and provides a prediction on whether the user is likely to have diabetes or not. The model was trained on a dataset of medical records, and the prediction is based on common features like glucose levels, BMI, and age.

The model uses Random Forest, an ensemble learning technique, to build a strong classifier by combining multiple decision trees. The app also includes a variety of model evaluation metrics and visualizations, including the confusion matrix, classification report, ROC curve, and correlation matrix.

Dataset Description
The dataset used in this project is based on diabetes patients and contains several health-related features. The data was originally collected from the Pima Indians Diabetes Database. It contains various medical attributes used to determine whether a person has diabetes.

Dataset Variables:
Pregnancies: The number of pregnancies the patient has had.
Glucose: The plasma glucose concentration after two hours in an oral glucose tolerance test.
BloodPressure: Diastolic blood pressure (mm Hg).
SkinThickness: Triceps skin fold thickness (mm).
Insulin: 2-Hour serum insulin (mu U/ml).
BMI (Body Mass Index): Weight in kg/(height in m)^2.
DiabetesPedigreeFunction: A function that represents the genetic relationship between the patient and their family history of diabetes.
Age: The age of the patient in years.
Target Variable:
Outcome: A binary variable where:
0: No diabetes
1: Diabetes
Approach
The Random Forest classifier was chosen as the primary model for predicting diabetes. It is an ensemble learning method that combines the predictions of multiple decision trees to improve the overall model performance.

Steps Taken:
Data Preprocessing:

The dataset was cleaned, and missing values were handled.
The features were scaled using the StandardScaler to normalize them and ensure that the model could effectively process the data.
Model Training:

A Random Forest classifier was used for training the model on the dataset. Random Forest was selected because of its ability to handle complex relationships in data and its robustness to overfitting.
Model Evaluation:

The model's performance was evaluated using several metrics:
Confusion Matrix: To visualize the true positives, false positives, true negatives, and false negatives.
Classification Report: To calculate precision, recall, F1-score, and support for both classes (diabetes and no diabetes).
ROC Curve: To evaluate the model's ability to discriminate between the two classes.
Correlation Matrix: To check for correlations between features in the dataset.
Streamlit Web App:

A Streamlit app was built to interactively predict whether a person has diabetes based on input values.
The app also displays the confusion matrix, ROC curve, and classification report for model evaluation.
