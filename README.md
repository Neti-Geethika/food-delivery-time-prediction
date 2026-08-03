# 🛵 Smart Food Delivery ETA Prediction System

An end-to-end machine learning project that predicts food delivery times using distance, weather, traffic, order timing, and courier experience — helping a delivery platform give customers an accurate ETA.

## Problem Statement

Food delivery platforms (like Swiggy, Zomato, Uber Eats) need to estimate delivery time accurately to improve customer satisfaction, reduce complaints, and assign delivery partners efficiently. This project builds a regression model that predicts delivery time (in minutes) from order and environmental conditions.

## Dataset

[Food Delivery Time Prediction (Kaggle)](https://www.kaggle.com/datasets/denkuznetz/food-delivery-time-prediction) — 1,000 delivery records with the following features:

| Feature | Description |
|---|---|
| Distance_km | Distance between restaurant and customer |
| Weather | Weather condition during delivery |
| Traffic_Level | Traffic congestion level |
| Time_of_Day | Time slot of the order |
| Vehicle_Type | Delivery vehicle used |
| Preparation_Time_min | Time taken to prepare the order |
| Courier_Experience_yrs | Delivery partner's experience |
| **Delivery_Time_min** | **Target** — actual delivery time |

## Approach

1. **Data Cleaning** — handled missing values (median/mode imputation), capped outliers using IQR
2. **EDA** — analyzed how distance, traffic, and weather relate to delivery time
3. **Feature Engineering** — added peak-hour flag, weather/traffic severity scores, distance-per-prep-minute ratio
4. **Preprocessing** — one-hot encoded categorical variables, scaled numeric features, 80/20 train-test split
5. **Model Comparison** — trained and compared 5 regression models
6. **Hyperparameter Tuning** — tuned the top model with `RandomizedSearchCV`
7. **Explainability** — used SHAP to identify which features drive predictions
8. **Deployment** — built an interactive Streamlit app for real-time predictions

## Results

| Model | MAE (min) | RMSE (min) | R² |
|---|---|---|---|
| **Linear Regression** | **5.83** | **8.67** | **0.831** |
| Gradient Boosting | 6.49 | 9.23 | 0.808 |
| XGBoost (tuned) | 6.22 | 8.94 | 0.820 |
| Random Forest | 6.59 | 9.51 | 0.796 |
| Decision Tree | 10.41 | 14.90 | 0.500 |

**Key finding:** Linear Regression outperformed every tree-based ensemble, including tuned XGBoost. This suggests the relationship between delivery time and its drivers (distance, traffic, weather) is largely linear in this dataset — the added complexity of boosting/bagging didn't translate into better generalization. Rather than defaulting to the most complex model, the simplest model that performs best was selected for deployment.

## Explainability (SHAP)

SHAP analysis on the tuned model showed **distance and traffic level** as the strongest drivers of delivery time, followed by weather severity and preparation time. This confirms the model's predictions align with real-world delivery logic rather than spurious correlations.

## Live App

🔗 *[Add your Streamlit Cloud link here once deployed]*

Users can input distance, weather, traffic, vehicle type, and time of day to get an instant predicted delivery time.

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · Streamlit · Matplotlib/Seaborn

## Repository Structure

```
├── Food_Delivery_ETA_Prediction.ipynb   # Full analysis & model training notebook
├── Food_Delivery_Times.csv              # Dataset
├── app.py                               # Streamlit deployment app
├── delivery_time_model.pkl              # Trained model
├── scaler.pkl                           # Feature scaler
├── model_columns.pkl                    # Expected input columns for the model
└── requirements.txt                     # Dependencies
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

Neti Geethika
