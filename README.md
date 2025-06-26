# 🛡️ Multi-Perspective E-commerce Fraud Detection System

This project uses a combination of **supervised and unsupervised machine learning models** to detect fraudulent transactions in an e-commerce setting. It explores multiple perspectives — such as Isolation Forest, One-Class SVM, XGBoost — to identify anomalies and predict transaction fraud risk. A **Streamlit app** is also integrated for interactive predictions and visualizations.

---

## 📌 Project Overview

- **Domain**: E-commerce, Cybersecurity
- **Goal**: Detect fraudulent transactions using machine learning
- **Dataset**: Realistic transaction logs with features like date, amount, SKU, product type, and labels (Fraud/Normal)
- **Models Used**:
  - Isolation Forest
  - One-Class SVM
  - XGBoost Classifier
- **Interface**: Streamlit web app (optional)

---

## 🧠 Key Features

- Preprocessing includes:
  - Date feature extraction (`day`, `month`, `weekday`, `is_weekend`)
  - Log-transformation of amount
  - Label Encoding for categorical columns
- Multiple classifiers trained and evaluated
- Anomaly-based fraud scoring
- Visualization of:
  - Fraud vs Normal distributions
  - Risk levels
  - Fraud score histograms
  - Feature importances
  - Confusion matrices

---

## 📊 Sample Visualizations

- 📌 Prediction Count Plot  
- 🔍 Fraud Score Distribution by Risk  
- ⚠️ Risk Level Boxplots  


---



## Preview of the output
![Screenshot 2025-06-26 152055](https://github.com/user-attachments/assets/4c39942e-cccc-49ec-81d6-fe7c76382f36)


## 🚀 Streamlit App (Optional)

You can run the Streamlit dashboard using:

```bash
streamlit run app.py
 
