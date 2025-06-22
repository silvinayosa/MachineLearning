# Name: Silvina Yosa
# Student ID: 10890525

# Load neccessary module
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import numpy as np 
import time
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv('winequality-white.csv', delimiter=';')

##################################################################################
# Simple EDA
##################################################################################
# Show first 10 data
print(df.head(10))
print("#"*100)
print()

# Show the summary statistics
print(df.describe())
print("#"*100)
print()

# Show the amount of missing value
print(df.isnull().sum())

# Making a correlation heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

##################################################################################
# Preprocessing Data
##################################################################################
# Preparing the attributes and label
X = df.drop('quality', axis=1)
y = df['quality']

# Split the data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Checking the shape
print(X_train.shape) # (3918, 11)
print(y_train.shape) # (3918,)
print(X_test.shape) # (980, 11)
print(y_test.shape) # (980,)

##################################################################################
# Machine Learning Model 1 (Decision Tree Classifier)
##################################################################################
# accuracy: 0.613265306122449
dt = DecisionTreeClassifier(random_state=33)

# Train the model
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Evaluate the model performance
dt_accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Classifier Accuracy: {dt_accuracy}")

##################################################################################
# Machine Learning Model 2 (Random Forest Classifier)
##################################################################################
# accuracy: 0.6979591836734694
rf = RandomForestClassifier(random_state=33)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model performance
rf_accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Classifier Accuracy: {rf_accuracy}")

##################################################################################
# Machine Learning Model 3 (Logistic Regression)
##################################################################################
# accuracy: 0.5316326530612245
lr = LogisticRegression(max_iter=1000, random_state=33)
sc = StandardScaler() # to prevent error "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT."

X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Train the model
lr.fit(X_train_scaled, y_train)

# Make predictions
y_pred = lr.predict(X_test_scaled)

# Evaluate the model performance
lr_accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {lr_accuracy}")

##################################################################################
# Machine Learning Model 4 (SVC)
##################################################################################
# Accuracy: 0.44285714285714284
svc = SVC(random_state=33)

# Train the model
svc.fit(X_train, y_train)

# Make predictions
y_pred = svc.predict(X_test)

# Evaluate the model performance
svc_accuracy = accuracy_score(y_test, y_pred)
print(f"SVC Accuracy: {svc_accuracy}")

##################################################################################
# Machine Learning Model 5 (Naive Bayes)
##################################################################################
# Accuracy: 0.4387755102040816
nb = GaussianNB()

# Train the model
nb.fit(X_train, y_train)

# Make predictions
y_pred = nb.predict(X_test)

# Evaluate the model performance
nb_accuracy = accuracy_score(y_test, y_pred)
print(f"Naive Bayes Accuracy: {nb_accuracy}")

##################################################################################
# Machine Learning Model 6 (KNN)
##################################################################################
# Accuracy: 0.4826530612244898
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn_classifier.fit(X_train, y_train)

# Make predictions
y_pred = knn_classifier.predict(X_test)

# Evaluate the model performance
knn_accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {knn_accuracy}")

##################################################################################
# StreamlitAPP
##################################################################################
# Making a title
st.title('Assignment 2 - Wine Prediction')

# Showing dataframe
st.subheader('Training Data')
if st.checkbox('Show Dataframe'): st.write(df.head(10))

# Simple analysis about training dataframe
# Showing correlation matrix
st.subheader('Correlation Matrix')
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
st.pyplot(plt)
plt.clf()
# showing bar chart
st.subheader('Bar Chart for Wine Quality')
quality_counts = df['quality'].value_counts()
plt.bar(quality_counts.index, quality_counts.values)
plt.xlabel('Quality')
plt.ylabel('Count')
st.pyplot(plt)
plt.clf()

st.subheader('Machine Learning Model Accuracy')
# I will use 3 models with highest accuracy: DTC, RFC, and LR
model_list = ['Decision Tree Classifier', 'Random Forest Classifier', 'Logistic Regression']
# Showing the accuracy for each model
classifiers = st.selectbox("Choose The Machine Learning Model!", model_list)
if classifiers == 'Decision Tree Classifier':
    st.write("Classification accuracy: ", dt_accuracy)
elif classifiers == 'Random Forest Classifier':
    st.write("Classification accuracy: ", rf_accuracy)
elif classifiers == 'Logistic Regression':
    st.write('Classification accuracy: ', lr_accuracy)

# Let user input their own wine in sidebar
st.sidebar.header('User Input Features')
# Let user input features for prediction
user_input = {}
for feature in X.columns:
    user_input[feature] = st.sidebar.slider(f'{feature} Range', float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))
# Convert user input to DataFrame for prediction
user_df = pd.DataFrame([user_input])

# Select machine learning model to predict
input_model = st.sidebar.selectbox('Choose the model you want to use!', model_list)
# Show the prediction result on each model
if input_model == 'Decision Tree Classifier':
    st.sidebar.write("Wine quality: ", dt.predict(user_df))
elif input_model == 'Random Forest Classifier':
    st.sidebar.write("Wine quality: ", rf.predict(user_df))
elif input_model == 'Logistic Regression':
    st.sidebar.write('Wine quality: ', lr.predict(user_df))
