# Technical Spec Sheet — Business Analytics Project
## Optimal Pricing Prediction for Airbnb Listings in Paris

**Course**: Business Analytics Using Python — HEC Paris  
**Dataset**: Inside Airbnb — Paris  
**Date**: May 2026

---

## 1. Business Problem

### Context

Paris is one of the densest Airbnb markets in the world, with over 60,000 active listings. Setting a nightly price is a central challenge for hosts: pricing too high reduces occupancy, pricing too low leaves value on the table. Yet most hosts set their prices intuitively, without any systematic modelling of price determinants.

### Core Question

> **What are the key drivers of Airbnb listing prices in Paris, and can we build a reliable predictive model to help hosts set an optimal price?**

### Analytical Sub-questions

1. Which intrinsic property characteristics (type, capacity, amenities) have the greatest impact on price?
2. Does location (arrondissement, GPS coordinates) account for a significant share of price variance?
3. Do reputation signals (average rating, number of reviews, host seniority) influence the listed price?
4. Can we identify homogeneous listing segments (clustering) that correspond to distinct pricing strategies?
5. Which supervised model offers the best predictive performance (linear regression vs. non-linear models)?

---

## 2. Data

### Source

**Inside Airbnb — Paris**  
URL: `https://insideairbnb.com/get-the-data/`  
License: Creative Commons CC0 (public domain)

### Files to Use

| File | Description | Format |
|---|---|---|
| `listings.csv` (detailed) | ~74 columns, one row per active listing | CSV |
| `reviews.csv` | Raw text of guest reviews | CSV |
| `calendar.csv` | Availability and price day-by-day over 365 days | CSV |
| `neighbourhoods.geojson` | Polygon boundaries of Parisian arrondissements | GeoJSON |

> **Note**: Use the detailed version of `listings.csv` (not `listings_summary.csv`) to access text columns and fine-grained attributes.

### Key Variables

#### Target Variable

| Variable | Type | Description |
|---|---|---|
| `price` | Float (after stripping the `$` symbol) | Listed nightly price |

#### Features — Property Characteristics

| Variable | Type | Notes |
|---|---|---|
| `room_type` | Categorical | Entire home / Private room / Shared room / Hotel |
| `property_type` | Categorical | Apartment, house, studio… |
| `accommodates` | Integer | Maximum number of guests |
| `bedrooms` | Integer | Number of bedrooms (impute if missing) |
| `beds` | Integer | Number of beds |
| `bathrooms` | Float | Number of bathrooms |
| `amenities` | JSON string | List of amenities (Wi-Fi, kitchen, parking…) |

#### Features — Location

| Variable | Type | Notes |
|---|---|---|
| `neighbourhood_cleansed` | Categorical | Normalised arrondissement label |
| `latitude` / `longitude` | Float | GPS coordinates |

#### Features — Reputation & Host

| Variable | Type | Notes |
|---|---|---|
| `review_scores_rating` | Float | Overall rating (/5 or /100 depending on snapshot) |
| `review_scores_cleanliness` | Float | Cleanliness sub-score |
| `review_scores_location` | Float | Location sub-score |
| `number_of_reviews` | Integer | Total number of reviews |
| `host_is_superhost` | Boolean | Airbnb Superhost badge |
| `host_identity_verified` | Boolean | Host identity verification |
| `calculated_host_listings_count` | Integer | Total number of listings by the host |

#### Features — Booking Policy

| Variable | Type | Notes |
|---|---|---|
| `minimum_nights` | Integer | Minimum stay length |
| `instant_bookable` | Boolean | Instant booking enabled |
| `cancellation_policy` | Categorical | Flexible / Moderate / Strict |
| `availability_365` | Integer | Number of available days over the year |

---

## 3. Analytical Pipeline

### Step 1 — Data Loading & Exploration (Sessions 1–2)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("listings.csv")
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
df['price'] = df['price'].str.replace('[$,]', '', regex=True).astype(float)
```

**EDA deliverables:**
- Price distribution (histogram, log-scale)
- Correlation heatmap (numerical features vs. price)
- Boxplots of price by arrondissement and `room_type`
- Listing density map (optional: `folium` or `geopandas`)

### Step 2 — Feature Engineering

#### 2.1 Cleaning

- Drop or impute missing values (`bedrooms`, `bathrooms`, `review_scores_rating`)
- Filter price outliers (e.g. keep the 1st–99th percentile range)
- Parse the `amenities` column (JSON-like string) to create binary indicator features

```python
import ast

df['amenities_list'] = df['amenities'].apply(ast.literal_eval)
key_amenities = ['Wifi', 'Kitchen', 'Air conditioning', 'Elevator', 'Dishwasher']
for amenity in key_amenities:
    df[f'has_{amenity.lower().replace(" ", "_")}'] = df['amenities_list'].apply(lambda x: amenity in x)
```

#### 2.2 Encoding

- One-Hot Encoding: `room_type`, `neighbourhood_cleansed`, `cancellation_policy`
- Label Encoding: `host_is_superhost`, `instant_bookable`, `host_identity_verified`

#### 2.3 Derived Features

- `log_price`: log transformation of the target to normalise its distribution
- `amenities_count`: total number of listed amenities
- `reviews_per_month_imputed`: median-imputed if missing

### Step 3 — Feature Selection (Sessions 2–3)

```python
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5).fit(X_train_scaled, y_train)
selected_features = X.columns[lasso.coef_ != 0]
```

**Methods to apply:**
- Pearson correlation (univariate selection)
- Lasso regression (L1 regularisation, penalises redundant features)
- Random Forest feature importance (cross-validated selection check)

### Step 4 — Supervised Modelling (Sessions 2–4)

#### Baseline: Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

model_lr = LinearRegression()
scores = cross_val_score(model_lr, X, y, cv=5, scoring='r2')
```

#### Advanced Models

| Model | Library | Hyperparameters to Tune |
|---|---|---|
| Linear Regression | `sklearn` | — |
| Decision Tree | `sklearn` | `max_depth`, `min_samples_leaf` |
| Random Forest | `sklearn` | `n_estimators`, `max_depth`, `max_features` |
| XGBoost | `xgboost` | `learning_rate`, `n_estimators`, `max_depth`, `subsample` |

#### Cross-Validation & Metrics

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Metrics: RMSE, MAE, R²
```

**Comparison metrics:**
- R² (explained variance)
- RMSE (Root Mean Squared Error) — in euros, on the untransformed price
- MAE (Mean Absolute Error)

### Step 5 — Unsupervised Modelling (Session 5)

**Goal**: identify segments of listings with homogeneous price profiles.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

inertias = [KMeans(n_clusters=k, random_state=42).fit(X_scaled).inertia_ for k in range(2, 11)]
# Elbow method to select k
```

**Expected interpretation:**
- Profile each cluster (average price, property type, dominant arrondissement, characteristic amenities)
- Name the segments, e.g. "Budget Studio Outskirts", "Central Family Apartment", "Premium Haussmann Listing"

### Step 6 (Optional) — Sentiment Analysis on Reviews

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
```

**Problem**: classify reviews as positive / negative (proxy: rating ≥ 4.5 = positive).  
**Features**: TF-IDF on English review text.  
**Models**: Naïve Bayes, Logistic Regression.  
**Business value**: identify lexical themes associated with highly-rated listings.

### Step 7 — Pipeline & Hyperparameter Tuning (Session 6)

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(random_state=42))
])

param_grid = {
    'model__n_estimators': [100, 300],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.05, 0.1]
}

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
```

---

## 4. Deliverables

### 4.1 Structured Colab Notebook

Expected sections:

1. **Import & Setup** — library imports, Google Drive mount
2. **Data Loading & Overview** — shape, dtypes, `.info()`, summary statistics
3. **EDA & Visualisation** — commented charts, intermediate insights
4. **Feature Engineering** — cleaning, encoding, derived features
5. **Feature Selection** — Lasso, correlations, RF importance
6. **Supervised Learning** — training, CV, model comparison
7. **Unsupervised Learning** — K-Means, cluster visualisation, profiling
8. **(Optional) Text Analysis** — sentiment/NLP on reviews
9. **Pipeline & Tuning** — final pipeline, GridSearch, best model
10. **Conclusion & Business Implications** — actionable recommendations

### 4.2 Presentation (12 min)

Suggested structure (5–7 slides):

| Slide | Content |
|---|---|
| 1 | Business problem & motivation |
| 2 | Dataset & key variables |
| 3 | EDA: main visual insights |
| 4 | Models & performance comparison |
| 5 | Listing segments (clustering) |
| 6 | Business recommendations |

### 4.3 Written Report

Sections (no length requirement):

- **Introduction**: problem statement, stakes, data
- **Data**: dataset description, cleaning, descriptive statistics
- **Analysis & Results**: pipeline, model performance, clustering results
- **Discussion**: limitations, implications, recommendations for a Parisian host

---

## 5. Grading Criteria & Course Concept Coverage

| Course Concept | Implementation in This Project | Session |
|---|---|---|
| Data types & structures | Pandas DataFrame, cleaning, JSON `amenities` parsing | 1 |
| Data visualization | Histograms, boxplots, heatmaps, scatter plots | 2 |
| Linear regression | Price baseline, coefficient interpretation | 2 |
| Feature selection | Lasso, correlations, RF importance | 3 |
| Logistic regression / Naïve Bayes | Sentiment analysis on reviews | 3 |
| Bias-Variance tradeoff & CV | KFold 5, train/test R² comparison | 4 |
| Decision Tree / Random Forest / XGBoost | Non-linear model comparison | 4 |
| K-Means clustering | Listing segmentation | 5 |
| Pipelines & Hyperparameter tuning | `sklearn.Pipeline` + `GridSearchCV` | 6 |

---

## 6. Technical Environment

```
Python         ≥ 3.10
pandas         ≥ 2.0
numpy          ≥ 1.24
scikit-learn   ≥ 1.3
xgboost        ≥ 2.0
matplotlib     ≥ 3.7
seaborn        ≥ 0.12
nltk / spacy   (optional, for NLP)
folium         (optional, for mapping)
```

**Runtime**: Google Colab (CPU sufficient; no GPU required)  
**Data storage**: Google Drive mounted via `drive.mount('/content/drive')`

---

## 7. Suggested Timeline

| Step | Suggested Deadline |
|---|---|
| Download & initial exploration (EDA) | End of Week 1 |
| Feature engineering + baseline supervised models | End of Week 2 |
| Advanced models + clustering | End of Week 3 |
| Final pipeline + tuning | End of Week 4 |
| Report writing + presentation preparation | Week 5 |

---

*Spec sheet prepared for the Business Analytics Using Python course — Prof. Xitong LI — HEC Paris, 2026.*