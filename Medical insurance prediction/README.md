# Medical Insurance Cost Prediction 🏥

A regression project predicting medical insurance charges based on age, BMI, smoking status, number of children, and region — comparing Linear Regression against regularized and ensemble models.

## 📊 Dataset

- **Source:** [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) (`insurance.csv`)
- **Size:** 1,338 rows, 7 columns — no missing values, no duplicates
- **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- **Target:** `charges`

## 🔧 Approach

### 1. EDA
- Checked for nulls and duplicates (none found)
- Examined value counts for categorical columns (`sex`, `smoker`, `region`)
- Plotted the distribution of `charges` — right-skewed, so applied a `log1p` transform to the target before modeling

### 2. Preprocessing Pipeline
Built with `ColumnTransformer` for clean, reusable preprocessing:
- **Numeric features** (`age`, `bmi`, `children`) → `StandardScaler`
- **Categorical features** (`sex`, `smoker`, `region`) → `OneHotEncoder`

### 3. Modeling
Trained and compared four models, all wrapped in a `Pipeline` with the same preprocessing step:
- Linear Regression (with 5-fold cross-validation)
- Ridge Regression (α = 0.1)
- Lasso Regression (α = 0.01)
- Random Forest Regressor (100 estimators)

## 📈 Results

| Model | R² Score |
|-------|----------|
| Linear Regression (5-fold CV mean) | 76.4% |
| Ridge | 80.5% |
| Lasso | 79.7% |
| **Random Forest** | **84.6%** |

**Conclusion:** Random Forest Regressor gave the best results on this dataset, outperforming all linear-based models — suggesting the relationship between features (especially the smoking × BMI interaction) has some non-linearity that plain linear models can't fully capture without manual feature engineering.

## 🛠️ Tech Stack
- Python, pandas, numpy
- scikit-learn (Pipeline, ColumnTransformer, LinearRegression, Ridge, Lasso, RandomForestRegressor)
- seaborn, matplotlib

## 📝 Key Learnings
- Log-transforming a skewed target (`charges`) before training is essential for linear models to perform well
- Using `Pipeline` + `ColumnTransformer` keeps preprocessing and modeling cleanly separated and prevents data leakage between train/test
- Regularization (Ridge/Lasso) gave a moderate boost over plain linear regression
- A non-linear model (Random Forest) outperformed all linear variants — a sign that smoking status likely interacts with other features (like BMI) in ways a linear model can't capture on its own
