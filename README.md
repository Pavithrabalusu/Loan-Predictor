# 💰 Loan Predictor

A full-stack Machine Learning web application that predicts the likelihood of a loan being approved based on applicant details. Built using **Django**, **Streamlit**, and **XGBoost**, this project showcases end-to-end development: from backend APIs and ML model training to a simple, interactive frontend.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Framework](https://img.shields.io/badge/Framework-Django%20%7C%20Streamlit-orange?logo=django)
![ML](https://img.shields.io/badge/Model-XGBoost-brightgreen?logo=machinelearning)
![Status](https://img.shields.io/badge/Status-Active-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Features

- 🔍 Predicts loan approval based on applicant data
- 📋 Handles fields like income, employment, credit history, loan amount, etc.
- 💡 Machine Learning model trained using real-world data
- 🎛️ Interactive frontend built with Streamlit
- ⚙️ REST APIs powered by Django for future integration
- 📦 Clean project structure and `.gitignore` to exclude sensitive/local files

---

## 🛠️ Tech Stack

| Layer        | Technology Used                      |
|--------------|---------------------------------------|
| Frontend     | Streamlit                            |
| Backend      | Django, Django REST Framework        |
| ML/Modeling  | Scikit-learn, XGBoost, Pandas        |
| Data Storage | CSV, Pickle                          |
| Deployment   | GitHub (manual or Streamlit-ready)   |

---

## 🧠 ML Model Info

- Dataset: Public loan approval dataset with fields like income, gender, loan amount, etc.
- Preprocessing: Missing value imputation, label encoding
- Model Used: `XGBoostClassifier`
- Output: Model saved as `.pkl` and `.json` for backend & frontend use

---

## 📂 Project Structure

LoanPredictor/
├── loan_prediction_backend/ # Django backend
│ ├── settings.py, urls.py, etc.
│ └── prediction/ # App: models, views, serializers
│ ├── ml_model/ # Trained models, utils, data
│ └── migrations/
├── streamlit_app.py # Streamlit frontend interface
├── frontend/ # Static files (optional)
├── requirements.txt # Python dependencies
├── .gitignore # Files/folders excluded from Git
