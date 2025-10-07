# 🧬 Breast Cancer Prediction ML

An end-to-end machine learning project that predicts breast cancer diagnosis using multiple classification algorithms.  
The goal is to assist early detection and improve diagnostic accuracy through data-driven insights.

---

## 📘 Overview
This project explores the **Breast Cancer Wisconsin (Diagnostic) dataset**, applying several supervised learning algorithms to classify tumors as *malignant* or *benign*.  
The workflow includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and comparison.

---

## ⚙️ Tech Stack
- **Programming:** Python (Jupyter Notebook)
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Techniques:** Classification, Feature Scaling, Hyperparameter Tuning, Model Evaluation

---

## 🧠 ML Models Implemented
- Logistic Regression  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Gradient Boosting  

Each model was evaluated using:
- Accuracy  
- Precision, Recall, F1-Score  
- ROC-AUC  
- Confusion Matrix  

---

## 🔍 Project Workflow
1. **Data Preprocessing**  
   - Handled missing values and outliers.  
   - Scaled numerical features using StandardScaler.  
   - Encoded target labels (Malignant = 1, Benign = 0).  

2. **Exploratory Data Analysis (EDA)**  
   - Visualized class distribution and key features.  
   - Analyzed correlations using heatmaps and pair plots.  

3. **Model Training and Evaluation**  
   - Trained multiple classification algorithms.  
   - Compared models using AUC, accuracy, and confusion matrices.  

4. **Results & Insights**  
   - Random Forest and SVM achieved the highest diagnostic accuracy.  
   - Feature importance analysis highlighted *mean radius* and *mean texture* as critical predictors.  

---

## 📈 Key Results
| Model | Accuracy | F1 Score | ROC-AUC |
|-------|-----------|----------|----------|
| Logistic Regression | 96.1% | 0.95 | 0.98 |
| Random Forest | 97.8% | 0.97 | 0.99 |
| SVM | 97.4% | 0.96 | 0.99 |
| Gradient Boosting | 97.6% | 0.97 | 0.99 |

*(Metrics are approximate and depend on random seed variations.)*

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/k-rajmani2k/breast-cancer-prediction-ml.git
