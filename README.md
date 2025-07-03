# 💼 Employee Attrition Prediction System

🚀 A machine learning-powered web application to predict whether an employee is likely to leave the company. Built with **Python**, trained using multiple **ML algorithms**, and deployed via **Django** on **PythonAnywhere**.

---

## 🌟 Overview

Employee attrition — the loss of employees through resignation or termination — can impact organizational productivity and morale. This project leverages **machine learning** to analyze HR data and predict employee attrition, empowering HR teams with data-driven insights for retention strategies.

We trained and compared multiple models and deployed the best-performing one as a real-time prediction system through a Django web interface.

---

## 🎥 Demo

- 📺 **Watch the demo video here:** [Demo Video Link](https://drive.google.com/file/d/13KXbov-zxXPtZmw0Sup_dgtPj-dE38qx/view?usp=sharing)
- 🌐 **Live Website:** [Visit Web App](https://souvikh007kv.pythonanywhere.com/)

---

## 📊 Problem Statement

Predict whether an employee will leave the company based on personal and workplace-related features using machine learning.

---

## 📌 Features

✅ Analyze employee records  
✅ Compare multiple ML models  
✅ **Random Forest** used as final model  
✅ Interactive **Django-based** frontend  
✅ Real-time prediction  
✅ Deployed on **PythonAnywhere**

---

## 🧠 Machine Learning Pipeline

- **Dataset:** IBM HR Analytics Employee Attrition & Performance dataset
- **Preprocessing:** Label encoding, One-hot encoding

### Models Evaluated:
- Logistic Regression
- Decision Tree
- **Random Forest** ✅
- SVC
- KNN
- AdaBoost

### Best Model:
`RandomForestClassifier`

### Evaluation Metrics:
- Accuracy  
- Classification Report  
- Confusion Matrix  
- ROC-AUC

**Saved Model:** `employee_attrition_model.joblib`

---

## 📁 Project Structure

```bash
employee-attrition-prediction/
│
├── employee_attrition_prediction.py
├── README.md
├── requirements.txt
└── Employee Attrition Prediction by Machine Learning (Report).pdf  # 📑 Detailed project report
```
## ⚙️ How to Run Locally

### 🔧 Prerequisites

- Python 3.x  
- pip  
- virtualenv  
---
## 📈 Sample Prediction Flow

1. User enters employee details in the frontend form.  
2. Data is preprocessed and fed into the trained model.  
3. Model predicts if the employee is at risk of attrition.  
4. Result is shown on the web page.

---

## 🧾 Resources

- 📂 **Dataset:** [IBM HR Analytics Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/ibm-attrition-dataset) 
- 📄 **Report:** See in the repo  
- 📺 **Demo Video:** [Watch Here](https://drive.google.com/file/d/13KXbov-zxXPtZmw0Sup_dgtPj-dE38qx/view?usp=sharing)

