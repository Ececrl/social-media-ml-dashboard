# 📊 Social Media Engagement & Sentiment Analytics Dashboard

An end-to-end Machine Learning and Business Intelligence (BI) interactive dashboard built with **Python** and **Streamlit**. 

This project transitions static ML reports into a dynamic, production-grade sandbox where stakeholders can simulate engagement predictions, test hyperparameters, and analyze text sentiment in real time.

---

## 🚀 Key Features

- **📊 Dynamic Performance KPIs:** Real-time visibility into regression metrics ($R^2$, MAE, RMSE) and Classification Accuracy.
- **🎛️ Interactive Engagement Simulator:** Predict engagement scores dynamically using tuned lag features, rolling averages, follower counts, and sentiment scores.
- **🧠 Live Sentiment Analysis (ANN):** Text classification engine powered by an Artificial Neural Network (MLPClassifier) reaching **96% accuracy**.
- **📈 Time-Series Analysis:** Actual vs. Predicted trend tracking over time.
- **🧩 Confusion Matrix & Feature Importance:** Visual insights into model decision-making and error distributions.
- **📄 Automated Report Generation:** One-click download for comprehensive metric summaries (`.txt`).

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.x
- **Dashboard & UI:** Streamlit
- **Machine Learning:** Scikit-Learn (Linear Regression, Ridge, Random Forest, MLPClassifier)
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn

---

## 🎯 Model Optimization & Feature Engineering

Initial models were significantly improved through advanced Feature Engineering:
1. **Lag Variables (`Lag_1`, `Lag_2`):** Captured historical engagement memory.
2. **Rolling Mean (`Rolling_Mean_3`):** Smoothed out short-term variance to capture macro-level trends.
3. **Scaling & Preprocessing:** Reduced feature dominance and boosted $R^2$ performance from baseline levels up to **94.47%**.

### Model Metrics Summary

| Model | Task | Key Metric | Score |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Engagement Regression | $R^2$ Score | **94.47%** |
| **Ridge Regression** | Engagement Regression | MAE | **1,337.61** |
| **Random Forest** | Engagement Regression | $R^2$ Score | **83.51%** |
| **MLPClassifier (ANN)** | Sentiment Classification | Accuracy | **96.00%** |

---

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Ececrl/social-media-ml-dashboard.git](https://github.com/Ececrl/social-media-ml-dashboard.git)
   cd social-media-ml-dashboard
   ---

## 📂 Data Source

The dataset used in this project was obtained from **[Kaggle](https://www.kaggle.com/)**. It contains social media metrics, engagement indicators, and text sentiment categories used for training and evaluating the machine learning models.