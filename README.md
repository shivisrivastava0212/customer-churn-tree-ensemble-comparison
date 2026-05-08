# 🌲 Customer Churn — Tree Model Battle

## 📋 Project Overview
This project implements a high-performance classification pipeline to predict customer attrition in the telecommunications industry. By comparing **Decision Trees**, **Random Forests**, and **XGBoost**, this repository demonstrates the strategic navigation of the **Bias-Variance Tradeoff** to achieve a production-ready model.

The project follows a **Golden Training Pipeline**, ensuring stability and industry-grade evaluation through gradient boosting and automated early stopping.

---

## 🏗️ Systematic Workflow

### 1. Data Processing & Feature Engineering
* **Brute Force Cleaning:** Implemented a custom utility to handle missing `TotalCharges` values and convert categorical strings into numeric formats using **One-Hot Encoding**.
* **Stratified Split:** Utilized a **stratified 80-20 train-test split** to maintain class proportions, ensuring the model is trained on a representative sample of churned vs. loyal customers.
* **Validation Environment:** Employed a 10% validation subset to monitor real-time learning and trigger advanced callbacks.

### 2. Industry-Grade Training Utilities (The "Battle")
To ensure the model reaches the **best weights**, the following ensemble strategies were evaluated:
* **Baseline (Decision Tree):** Established a performance floor, identifying initial high-variance risks.
* **Bagging (Random Forest):** Reduced variance by averaging 100 independent trees to improve generalization.
* **Boosting (XGBoost):** Optimized for **Bias Reduction** by sequentially correcting errors from previous iterations.
* **Early Stopping:** Monitored `logloss` with a patience of 10 rounds to prevent the model from entering the "overfitting zone."

---

## 📊 Performance Metrics
The models were evaluated using a comprehensive suite of metrics, with **XGBoost** emerging as the champion:

| Metric | Decision Tree | Random Forest | **XGBoost (Winner)** |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 80% | 79% | **81.0%** |
| **Precision** | 0.79 | 0.77 | **0.79** |
| **Recall** | 0.46 | 0.51 | **0.55** |
| **F1-Score** | 0.58 | 0.61 | **0.65** |

### Visual Insights

**Confusion Matrix**
The matrix highlights the model's ability to balance True Negatives while capturing the minority "Churn" class effectively.

**Training Stability (Learning Curve)**
The **Log Loss** graph confirms that the model successfully converged, with **Early Stopping** halting training at the optimal iteration to maximize generalization.

---

## ⚙️ Hyperparameters & Configuration
Following the **Industry Checklist**, these parameters provided the most stable results for XGBoost:
* **Learning Rate:** 0.05 (Optimized for gradual convergence).
* **Max Depth:** 6 (Balanced to prevent tree complexity from causing overfitting).
* **Early Stopping:** Active (Restored **Best Iteration** via `best_iteration` attribute).
* **Best Model Weights:** Saved in `.json` format for cross-platform deployment.

---

## 📉 Overfitting Control (Bias vs. Variance)
To maintain a robust balance, the following techniques were implemented:
* **Ensemble Averaging:** Used Random Forest to mitigate the high variance of single decision trees.
* **Sequential Correction:** Used Gradient Boosting (XGBoost) to reduce bias by focusing on difficult-to-predict residuals.
* **Validation Monitoring:** Stopped training at the exact point where validation loss plateaued, ensuring the model remains valid for unseen data.

---

## 🚀 How to Run & Deploy

### Option 1: Google Colab (Recommended)
1. Upload the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file.
2. Run the notebook cells to trigger the **Model Battle**.
3. Download the generated `best_churn_model.json`.

### Option 2: Local Execution
```bash
git clone [https://github.com/shivisrivastava0212/customer-churn-tree-ensemble-comparison.git](https://github.com/shivisrivastava0212/customer-churn-tree-ensemble-comparison.git)
pip install pandas scikit-learn xgboost matplotlib seaborn
python scripts/churn_pipeline.py
