# 🌲 Customer Churn — Tree Model Battle

## 📋 Project Overview
This project implements a high-performance classification pipeline to predict customer attrition in the telecommunications industry. By comparing **Decision Trees**, **Random Forests**, and **XGBoost**, this repository demonstrates the strategic navigation of the **Bias-Variance Tradeoff** to achieve a production-ready model.

The project follows a **Golden Training Pipeline**, ensuring stability and industry-grade evaluation through gradient boosting and automated early stopping.

---

## 🏗️ Systematic Workflow

### 1. Data Processing & Brute Force Cleaning
* **Standardization:** While tree-based models are robust to feature scales, I ensured data consistency through automated cleaning utilities.
* **Brute Force Cleaning:** Implemented a custom utility to handle missing `TotalCharges` values and convert categorical features into numeric formats via One-Hot Encoding.
* **Stratified Split:** Utilized a **stratified 80-20 train-test split** to maintain class proportions, ensuring a representative sample of churned vs. loyal customers.
* **Evaluation Environment:** Leveraged **Google Colab** with hardware acceleration to manage model iterations and scaling efficiently.

### 2. Industry-Grade Training Utilities
To ensure the model reaches the **best weights**, the following "Battle" parameters were implemented:
* **Baseline (Decision Tree):** Established initial performance benchmarks and identified high-variance risks.
* **Bagging (Random Forest):** Reduced variance by averaging 100 independent trees to improve generalization.
* **Boosting (XGBoost):** Optimized for **Bias Reduction** by sequentially correcting errors from previous iterations.
* **Early Stopping:** Monitored `val_loss` with a patience of 10 rounds to prevent the model from entering the "overfitting zone."

---

## 📊 Performance Metrics (Final Results)
The model was evaluated using a comprehensive suite of industry-standard metrics, with **XGBoost** emerging as the champion:

| Metric | Decision Tree | Random Forest | **XGBoost (Winner)** |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 80.62% | 78.92% | **80.55%** |
| **Precision (Churn)** | 0.70 | 0.64 | **0.67** |
| **Recall (Churn)** | 0.46 | 0.46 | **0.55** |
| **F1-Score (Churn)** | 0.56 | 0.54 | **0.60** |

### Visual Insights

**Confusion Matrix (Churn Prediction)**
<img width="511" height="395" alt="Screenshot 2026-05-09 at 3 39 26 PM" src="https://github.com/user-attachments/assets/14fb698a-056f-49b3-9e4c-5c36fcaf729a" />

The matrix shows strong performance in identifying loyal customers, accurately capturing **196 potential churners** (Recall) while maintaining high precision.

**Learning Curve (Early Stopping)**
<img width="853" height="462" alt="Screenshot 2026-05-09 at 3 39 46 PM" src="https://github.com/user-attachments/assets/0313b29b-8a1a-4895-bd1f-c5b8a2a311b8" />

The **Log Loss** graph confirms training stability. The model reached its optimal state at **Iteration 79**, where the validation loss reached its minimum before overfitting could occur.

---

## ⚙️ Hyperparameters & Configuration
Following the **Industry Checklist**, these parameters provided the most stable results:
* **Optimizer/Algorithm:** Gradient Boosted Decision Trees (XGBoost).
* **Learning Rate:** 0.05 (Optimized for gradual and stable convergence).
* **Max Depth:** 6 (Balanced to prevent tree complexity from causing overfitting).
* **Early Stopping:** Active (Stopped and restored weights from **Best Iteration: 79**).
* **Loss Function:** Logarithmic Loss (**Log Loss**), the industry standard for binary classification probability optimization.

---

## 📉 Training Stability & Overfitting Control
To maintain a balance between **Bias and Variance**, I implemented the following techniques:
* **Bagging:** Used Random Forest to mitigate the high variance inherent in single decision trees.
* **Boosting:** Used XGBoost to reduce bias by focusing on difficult-to-predict residual errors.
* **Early Stopping:** Monitored validation loss and restored **best model weights** from the optimal epoch to ensure the model did not train into the "overfitting zone."

---

## 🚀 How to Run & Deploy

## Google Colab (Recommended)
1. Upload the `Telco Customer Churn.csv` file when prompted.
2. The pipeline will automatically clean data, train models, and generate the evaluation report.

---
## 🧠 Lessons Learned
* **Bias vs. Variance Tradeoff**: By comparing different tree architectures, I observed how XGBoost reduces bias more effectively than a standard Random Forest for this specific dataset.

* **The Power of Early Stopping**: Implementing monitoring ensured that the final model utilized the Best Weights from Iteration 79, rather than just the final iteration.

* **Metric Selection**: I learned that for churn prediction, Recall is more crucial than Accuracy; identifying customers who are about to leave (True Positives) allows the business to take proactive retention steps.

* **Model Serialization**: Exporting weights as .json and .pkl ensures the model is "deployment-ready" for real-time inference in a production environment.

----
👨‍💻 Author
Shivi Srivastava Aspiring AI Engineer | Amity University Uttar Pradesh
[LinkedIn Profile] (https://www.linkedin.com/in/shivi-srivastava-8a5086310/)| [GitHub Portfolio] (https://github.com/shivisrivastava0212)

