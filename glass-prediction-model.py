# Name: Silvina Yosa
# Student ID: 10890525

# import the necessary modules
import pandas as pd
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import joblib

####################################################################
# Part 1: Load a dataset and Look at the summary of the dataset
####################################################################
# Load the dataset into dataframe
df = pd.read_csv('./dataset/glass.csv', index_col='Id_number')

# Print first five rows of data
print("First five rows of data")
print(df.head(5))
print("#"*50)
print()

# Print the data shape
print("Data shape: {}.".format(df.shape))
print("#"*50)
print()

# Print the data columns value
print("Data columns value")
print(df.columns)
print("#"*50)
print()

# Print the data types of each columns and the other stuff
print("More info about the data")
print(df.info())
# print(df.dtype) to only see the data type of each coumn
print("#"*50)
print()

# Print basic statistics of all numerical columns
print("Basic statistics for all numerical column")
print(df.describe())
print("#"*50)
print()

# check if there is any missing value
# print(df.isnull().sum()) # 0 for every columns
####################################################################
# Part 2: EDA (Exploratory Data Analysis) of the dataset
####################################################################
# a. Bar Chart:
df.plot.bar(x='Type_of_glass')
plt.title('Type of Glass and other attributes')
plt.show()

# b. Histogram: distribution of class attributes: Type_of_glass
sns.displot(df['Type_of_glass'], bins=6)
plt.xlabel('Type of Glass')
plt.ylabel('Distribution')
plt.title('Type of Glass Distribution Histogram')
plt.show()

# c. Box Plot: the IQR (interquartile range) of features and class attributes
for i in df.columns[:-1]:
    sns.boxplot(df,x='Type_of_glass',y=i)
    plt.title('Boxplot of type of glass and {}'.format(i))
    plt.xlabel('Type of glass')
    plt.ylabel(i)
    plt.show()

# d: Scatter Plot: show different colors for different types of glass (e.g. Si vs. Type_of_class, Ca vs. Type_of_class, etc)
for i in df.columns[:-1]:
    plt.scatter(df['Type_of_glass'],df[i])
plt.xlabel('Type_of_glass')
plt.ylabel('Other variables')
plt.title('Scatter plot for type of glass and other variables')
plt.show()

# e: Correlation Matrix: correlations among the features and class attributes
print("Correlation Matrix")
print(df.corr())
print("#"*50)
print()

# f: Heat Map: correlations among the features and class attributes. 
sns.heatmap(df.corr())
plt.title('Correlation matrix')
plt.show()

####################################################################
# Part 3: Train and Build Machine Learning Models
####################################################################
# Defining x and y
# x: input (attributes to predict)
# y: output (label to be predicted)
Y = np.asarray(df['Type_of_glass'])
X = np.asarray(df.drop('Type_of_glass', axis=1))

# 1. Logistic Regresion Method
train_lr, test_lr = train_test_split(df, test_size=0.3, random_state=42)
y_train_lr = train_lr['Type_of_glass']
y_test_lr = test_lr['Type_of_glass']
x_train_lr = train_lr.drop('Type_of_glass', axis=1)
x_test_lr = test_lr.drop('Type_of_glass', axis=1)
lrModel = LogisticRegression(solver='saga')
lrModel.fit(x_train_lr, y_train_lr)

# 2. Naive Bayes Method
X_train_naive, X_test_naive = train_test_split(df, test_size=0.3, random_state=42)
X_naive = ['RI','Na','Mg','Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
y_naive = ['Type_of_glass']
gaussian_NB = GaussianNB()
gaussian_NB.fit(X_train_naive[X_naive].values, X_train_naive['Type_of_glass'])

# 3. KNN Classifier
# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5) 
####################################################################
# Part 4: Model Evaluation
####################################################################
# 1. Logistic Regression Model
print('Logistic accuracy score: {}.'.format(lrModel.score(x_test_lr,y_test_lr)))
y_pred_lr = lrModel.predict(x_test_lr)
print('Logistic Regression predicted value')
print(y_pred_lr)
print("#"*50)
df_comparison_lr = pd.DataFrame(
    {
        'Real_Values': y_test_lr,
        'Predicted_Values': y_pred_lr
    }
)
print()
print(df_comparison_lr)
print("#"*50)
print()

# Naive bayes
y_pred_naive = gaussian_NB.predict(X_test_naive[X_naive])
print('Predicted value naive')
print(y_pred_naive)
print("#"*50)
print()

print("Number of mislabeled points out of a total {} points: {}, \
    prediction accuracy: {}".format(X_test_naive.shape[0],\
    (X_test_naive['Type_of_glass'] != y_pred_naive).sum(), \
        100 * (1-(X_test_naive['Type_of_glass']!=y_pred_naive).sum() / X_test_naive.shape[0])))

print("#"*50)
print()

# KNN Classifier
knn.fit(X_train, y_train)
y_pred_knn =knn.predict(X_test)
print('Prediction for KNN classifier')
print(y_pred_knn)
print("#"*50)
print()
print('KNN model accuracy with k=5 is {}'.format(format(metrics.accuracy_score(y_test,y_pred_knn))))
print("#"*50)
print()

####################################################################
# Part 5: Make predictions
####################################################################
# we dump knn model as it has the higher score this far...
joblib.dump(knn, "Type_of_glass_prediction.pkl")

# Then we load it to be used in different example
knn2 = joblib.load("Type_of_glass_prediction.pkl")

# dictionary containing type of glass
Type_of_glass={
    1:'building_windows_float_processed',
    2:'building_windows_non_float_processed',
    3:'vehicle_windows_float_processed',
    4:'vehicle_windows_non_float_processed',
    5:'containers',
    6:'tableware',
    7:'headlamps'
}

print('for example, we have this number of input')
predictions = knn2.predict([[1.51756,13.15,3.61,1.05,73.24,0.57,8.24,0,0]])
print('The predicted type of glass is {}'.format(Type_of_glass[predictions[0]]))
    