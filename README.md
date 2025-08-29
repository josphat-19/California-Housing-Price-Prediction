# California Housing Price Prediction

A complete end-to-end machine learning project to predict median house prices in California districts based on the 1990 census data.  
This project demonstrates the full ML lifecycle — from **data exploration and cleaning** to **model training, evaluation, and preparation for deployment**.

---

## 📌 Project Overview
The goal of this project is to build a robust machine learning model that can accurately predict the median house value for any district in California.  

The project covers:
- Performing **Exploratory Data Analysis (EDA)** to understand the data.
- Engineering new features to improve model performance.
- Building a **scalable preprocessing pipeline**.
- Training and comparing multiple regression models.
- **Fine-tuning** the best model using GridSearchCV.
- Evaluating the final model on a held-out test set.
- Packaging the final model for production use.

---

## 📂 Dataset
- **Source**: The dataset is commonly known as the *California Housing dataset* (1990 U.S. Census).
- **Unit of analysis**: Each row represents a **census block group** (district) with a population of 600–3,000 people.

**Features:**
- `longitude` → Longitude of the district  
- `latitude` → Latitude of the district  
- `housing_median_age` → Median age of houses  
- `total_rooms` → Total number of rooms  
- `total_bedrooms` → Total number of bedrooms  
- `population` → Total population  
- `households` → Total households  
- `median_income` → Median household income (in tens of thousands of USD)  
- `ocean_proximity` → Categorical: proximity to ocean (`<1H OCEAN`, `INLAND`, etc.)  
- `median_house_value` → **Target variable**: Median house value (USD)  

---

## 📁 Project Structure
```
California-Housing-Price-Prediction/
│
├── data/                               # Raw dataset
│   ├── housing.csv
│
├── notebook/                           
│   ├── Housing.ipynb                   # Main Jupyter notebook with analysis                        
│
├── models/                             # Directory for saved models                             
│   ├── final_model.pkl                 # Best model pipeline (joblib)
|
├── README.md                           # Project overview and instructions
├── LICENSE                             # License information for the repository
└── requirements.txt                    # Dependencies and requirements for the project
```

---

## 🔑 Key Steps and Methodology

### 1. Data Exploration & Analysis (EDA)
- Inspected schema, data types, missing values (`total_bedrooms`).
- Visualized distributions of numerical features with histograms.
- Identified **median_income** as the strongest predictor.
- Detected **capped values** for `median_house_value` and `median_income`.

### 2. Data Preprocessing Pipeline
- Built using **ColumnTransformer** and **Pipeline**.
- **Numerical features**: median imputation + scaling (`StandardScaler`).  
- **Categorical features**: one-hot encoding of `ocean_proximity`.  
- **Feature engineering**:  
  - `rooms_per_household`  
  - `population_per_household`  
  - `bedrooms_per_room`

### 3. Model Training & Selection
- Trained multiple regressors:
  - `LinearRegression` (baseline)
  - `DecisionTreeRegressor` (overfit)
  - `RandomForestRegressor` (best performer)
- Used **10-fold cross-validation** for reliable evaluation.

### 4. Hyperparameter Tuning
- Applied **GridSearchCV** on `RandomForestRegressor`.
- Tuned `n_estimators`, `max_features`, and bootstrap options.
- Achieved significant performance improvement over defaults.

### 5. Final Evaluation
- Evaluated the **best-tuned RandomForest model** on the held-out test set.
- Reported **RMSE** as the key metric.
- Computed a **95% confidence interval** for the generalization error.

---

## 📊 Results
- **Final Model**: Tuned RandomForestRegressor  
- **Test RMSE**: ~ **$47,000**  

This means predictions are typically within **$47k** of actual median house values.  
Considering house prices ranged from **$14,999 to $500,000**, the model demonstrates solid predictive performance.

**Key Insights:**
- `median_income` is the strongest predictor.  
- Engineered feature `bedrooms_per_room` is highly important.  
- Location features (`longitude`, `latitude`, `ocean_proximity`) are critical.

