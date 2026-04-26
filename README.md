# TASK 2
# Unemployment Rate Prediction using Machine Learning



##  Project Overview

This project focuses on building a **machine learning model to predict unemployment rates** using historical economic data. The model captures **temporal patterns, regional variations, and economic trends** to make accurate predictions.

Unemployment is a critical economic indicator, and predicting it can support:

* Policy-making decisions
* Economic analysis
* Workforce planning



##  Objective

The goal of this project is to:

* Predict **Estimated Unemployment Rate (%)**
* Capture **time-based patterns (seasonality & trends)**
* Handle **regional and demographic variations**
* Build a model that **generalizes well on unseen data**



##  Dataset Description

The dataset contains the following features:

| Feature                                 | Description                        |
| --------------------------------------- | ---------------------------------- |
| Date                                    | Time of observation                |
| Region                                  | State-wise data                    |
| Area                                    | Urban / Rural classification       |
| Estimated Unemployment Rate (%)         | Target variable                    |
| Estimated Employed                      | Number of employed individuals     |
| Estimated Labour Participation Rate (%) | Workforce participation            |
| Frequency                               | Data frequency (Monthly/Quarterly) |


##  Data Preprocessing

* Cleaned column names
* Converted `Date` to datetime format
* Sorted data for time-series consistency
* Removed unnecessary columns (`Frequency`)
* Handled missing values



##  Feature Engineering (Key Highlight)

To improve model performance, several advanced features were created:

### 🔹 Time Features

* Year
* Month

### 🔹 Cyclical Encoding

```python
Month_sin = sin(2π * Month / 12)
Month_cos = cos(2π * Month / 12)
```

✔ Captures seasonality



### 🔹 Lag Features (Time Dependency)

```python
lag_1, lag_2, lag_3
```

✔ Helps model learn from past unemployment values

---

### 🔹 Rolling Features

```python
rolling_mean_3
rolling_std_3
```

✔ Captures trends and volatility



### 🔹 Categorical Encoding

* Area → Binary encoding (Urban = 1, Rural = 0)



##  Model Selection

### 🔸 Models Tested

* Linear Regression (baseline)
* Random Forest Regressor
* XGBoost Regressor

### 🔸 Final Model

✅ **Random Forest Regressor**


#### Why Random Forest?
The Random Forest Regressor was used as the final model for predicting unemployment rates due to its ability to handle non-linear relationships, feature interactions, and noisy real-world data.

Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting.

* Handles non-linear relationships
* Robust to outliers
* Works well with tabular data
* Requires less tuning compared to boosting models



##  Model Training

* Used **time-based train-test split (80/20)**
* Avoided data leakage
* Tuned hyperparameters:

```python
 RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    random_state=42
)

```



##  Model Evaluation

### 🔹 Metrics Used

* **R² Score** → Measures model performance
* **MAE (Mean Absolute Error)** → Average error
* **RMSE** → Penalizes large errors




### 🔹 Results

| Metric   | Value              |
| -------- | ------------       |
| R² Score | 0.9468741864974338 |
| MAE      | 1.3484478999095797 |
| RMSE     |2.1910535649242466  |




## 📈 Model Validation

### ✔ Random Sample Testing

Predictions were compared with actual values:

```
Predicted: 7.9189669156954166 | Actual: 7.79
Predicted:  2.0905445714285738 | Actual: 2.00
Predicted: 14.72761866666668 | Actual: 15.22
```

✔ Errors are minimal (< 0.6)



###  Visual Evaluation

* Actual vs Predicted scatter plot
* Error distribution analysis
 


##  Key Insights

* Unemployment shows **seasonal patterns**
* Past values (lag features) are highly predictive
* Regional differences significantly impact unemployment
* Ensemble models outperform linear models



##  Challenges Faced

* Handling sudden spikes (e.g., COVID impact)
* Avoiding overfitting
* Capturing time dependencies correctly
* Feature engineering complexity



##  Model Limitations

* Slight underestimation during extreme spikes
* Depends on historical patterns (no external factors included)



##  Future Improvements

* Use advanced models like:

  * XGBoost (fully tuned)
  * LSTM (deep learning for time series)
* Include external economic indicators (GDP, inflation)
* Deploy as a web app using Flask/Streamlit



##  Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* Random Forest



##  How to Run the Project

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook or script
```bash
jupyter notebook Unemployment_analysis.ipynb
```
##  Conclusion

This project successfully demonstrates how **machine learning can be applied to economic data** for predictive analysis. By leveraging **feature engineering and ensemble learning**, the model achieves strong performance and real-world applicability.



####  Author

Ummalla Prathibha
Data Science Intern


