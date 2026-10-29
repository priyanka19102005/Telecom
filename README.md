# 📡 Telecom Customer Churn Prediction & Retention Hub

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=priyanka19102005/Telecom&branch=main&mainModule=app.py)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

An end-to-end Machine Learning web application deployed with Streamlit that analyzes telecommunications customer behavioral patterns and predicts churn risk to guide customer retention strategies.

---

## 📌 Features

- **Real-Time Customer Churn Predictor**: Compute churn probabilities and receive automated retention playbooks (contract offers, tech support promotions, onboarding outreach).
- **Model Comparison**: Benchmarks Random Forest, Logistic Regression, and Decision Tree with ROC-AUC, Accuracy, Precision, Recall, F1, and Confusion Matrices.
- **Churn Driver Analytics**: Visualizes churn categories (Competitor, Dissatisfaction, Price), churn rate by contract, and financial loss metrics.
- **Batch Customer Scoring**: Upload raw CSV callsets, score hundreds of accounts with predicted churn probabilities and risk tiers, and download the annotated CSV.

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/priyanka19102005/Telecom.git
cd Telecom
pip install -r requirements.txt
streamlit run app.py
```
