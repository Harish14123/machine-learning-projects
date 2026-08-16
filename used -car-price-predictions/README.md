# Used Car Price Prediction 🚗

A linear regression project that predicts the resale price of used cars based on brand, mileage, engine specs, fuel type, and other features.

## 📊 Dataset

- **Source:** [Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset) by taeefnajib on Kaggle
- **Size:** 4,009 rows, 12 original columns
- **Target variable:** `price`

## 🔧 Approach

### 1. Data Cleaning
- Removed the `clean_title` column (mostly redundant/missing)
- Cleaned `milage` and `price` columns (stripped `$`, `,`, and units from strings, converted to numeric)
- Filled missing values in engineered numeric columns with the median

### 2. Feature Engineering
- Extracted `HP`, `Engine_Size`, and `Cylinders` from the raw `engine` text column using regex
- Created `car_age` from `model_year`
- Grouped `brand` into `brand_group` (Economy / Luxury / Exotic / Electric / Other)
- Bucketed `model` into top 30 most common models (`model_main`), rest labeled "Other"
- Simplified `fuel_type` and `transmission` into cleaner categories
- Bucketed exterior/interior colors into main colors + "Other"
- Encoded `accident` as binary (0/1)

### 3. Handling Skew & Outliers
- `price` had a skew of **19.5** (heavily right-skewed) — applied `log1p` transform
- Removed outliers using the IQR method on the transformed price
- Clipped `milage` and `HP` at the 1st/99th percentiles

### 4. Encoding
- One-hot encoded all categorical features with `drop_first=True` to avoid the dummy variable trap

### 5. Modeling
- Train/test split (80/20)
- Standard-scaled numeric features (fit on train only, applied to test — no data leakage)
- Trained a plain `LinearRegression` model

## 📈 Results

| Metric | Train | Test |
|--------|-------|------|
| R² | 0.829 | 0.828 |
| MAE (log scale) | 0.232 | 0.239 |
| RMSE (log scale) | 0.326 | 0.329 |

Train and test scores are nearly identical — no overfitting.

## 🛠️ Tech Stack
- Python, pandas, numpy
- scikit-learn (LinearRegression, StandardScaler, train_test_split)
- seaborn, matplotlib for visualization

## 📝 Key Learnings
- A heavily skewed target variable can wreck linear regression's R² — log-transforming `price` was the single biggest fix, taking R² from ~11% to ~83%
- High-cardinality categorical columns (like `model` with hundreds of unique values) need to be bucketed before one-hot encoding, or you end up with too many sparse, noisy features
- Always evaluate on a held-out test set, not just training data — training R² alone can hide underfitting or overfitting
