# Machine Learning Projects 🎓

A curated collection of small-scale machine learning experiments completed during my undergraduate studies.  
Each project focuses on classification models and exploratory data analysis (EDA), built using Python.  
Projects are organized into self-contained scripts or Streamlit apps, featuring clean visualizations and straightforward logic for learning and demonstration purposes.


---

## 🗂️ Repository Structure

MachineLearning/
├─ README.md
├─ wine-prediction/
│ ├─ wine_prediction.py
│ ├─ requirements.txt
│ └─ README.md
└─ glass-classifier/
├─ glass_classifier.py
├─ dataset/glass.csv
├─ requirements.txt
└─ README.md

---

## 📁 Project Descriptions

### 🔹 `wine-prediction/`

#### 🔍 Features

A machine learning app that predicts wine quality based on chemical properties. It includes:

- EDA (correlation matrix, bar charts)
- Multiple ML models: Decision Tree, Random Forest, Logistic Regression
- Interactive **Streamlit** interface for model comparison and custom predictions

#### 🛠 How to Run

```bash
cd wine-prediction
pip install -r requirements.txt
streamlit run wine_prediction.py
```

### 🔹 `glass-type-prediction/`

A classification project that predicts the **type of glass** based on its chemical composition using multiple machine learning algorithms.

#### 🔍 Features

- Exploratory Data Analysis (EDA):
  - Histograms, boxplots, scatter plots, and correlation heatmap
- Multiple classification models implemented:
  - Logistic Regression
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
- Model evaluation using accuracy, prediction comparison, and confusion matrix
- Model persistence using `joblib` (KNN saved as `.pkl`)

#### 🛠 How to Run

```bash
cd glass-type-prediction
pip install -r requirements.txt
python glass_prediction.py
```

### 🔹 `cardio-prediction/`

A machine learning project that predicts the risk of cardiovascular disease based on health-related features such as blood pressure, cholesterol, BMI, and more.

#### 🔍 Features

- Data preprocessing and EDA (age groupings, histograms, correlation heatmaps)
- Multiple classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors
  - Naive Bayes
- Model evaluation using accuracy scores and confusion matrix


#### 🛠 How to Run

```bash
cd cardio-prediction
pip install -r requirements.txt
python cardio_prediction.py
```

## 🧠 Goals
✅ Practice data preprocessing, visualization, and model training
✅ Compare classifier performance in simple datasets
✅ Build interactive demos using Streamlit
✅ Keep code clean, modular, and reproducible

## 🔧 Tools Used
Python

Pandas, NumPy

Scikit-learn

Seaborn, Matplotlib

Streamlit

Joblib
