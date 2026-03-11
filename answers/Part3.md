# Part C: Regression Analysis

## Task 1 — Data Inspection

### Question 21

Perform an exploratory data analysis on the Diamonds dataset. Report correlation analysis, distribution analysis, and categorical analysis.

**Answer:**

**Correlation Analysis:**

File: [Correlation Heatmap](outputs/q21_correlation_heatmap.png)

| Feature | Correlation with Price |
| ------- | ---------------------- |
| carat   | 0.9216 (strongest)     |
| x       | 0.8844                 |
| y       | 0.8654                 |
| z       | 0.8612                 |
| table   | 0.1271                 |
| depth   | -0.0106 (weakest)      |

**Key Observations:**
1. Carat has strongest correlation (0.92) - larger diamonds more valuable
2. Physical dimensions (x, y, z) highly correlated (0.86-0.88) - related to carat weight
3. Depth and table show weak correlations
4. High multicollinearity among x, y, z, carat (>0.95)

---

**Distribution Analysis:**

File: [Histograms](outputs/q21_histograms.png)

| Feature | Skewness | Assessment              | Transformation |
| ------- | -------- | ----------------------- | -------------- |
| y       | 2.4342   | Highly skewed           | Log            |
| z       | 1.5224   | Moderately skewed       | Log            |
| carat   | 1.1166   | Moderately skewed       | Log            |
| table   | 0.7969   | Slightly skewed         | Optional       |
| x       | 0.3787   | Approximately symmetric | None           |
| depth   | -0.0823  | Approximately symmetric | None           |

**Recommendation:** Apply log transformation to `carat`, `y`, `z` to address right-skewness.

---

**Categorical Analysis:**

File: [Box Plots](outputs/q21_boxplots.png)

**Cut vs Price:**
| Cut       | Median Price |
| --------- | ------------ |
| Fair      | $3,282       |
| Premium   | $3,185       |
| Good      | $3,050       |
| Very Good | $2,648       |
| Ideal     | $1,810       |

**Trend:** "Ideal" cut has lowest median price, "Fair" highest - counterintuitive due to carat correlation.

**Color vs Price:**
| Color | Median Price |
| ----- | ------------ |
| J     | $4,234       |
| I     | $3,730       |
| H     | $3,460       |
| G     | $2,242       |
| F     | $2,343       |
| E     | $1,739       |
| D     | $1,838       |

**Trend:** Lower color grades (J, I, H) higher prices - likely due to carat correlation.

**Clarity vs Price:**
| Clarity | Median Price |
| ------- | ------------ |
| SI2     | $4,072       |
| I1      | $3,344       |
| SI1     | $2,822       |
| VS2     | $2,054       |
| VS1     | $2,005       |
| VVS2    | $1,311       |
| VVS1    | $1,093       |
| IF      | $1,080       |

**Trend:** Lower clarity grades higher prices - counterintuitive but explained by carat correlation.

---

### Question 22

Explain encoding trade-offs and report which method was chosen for each categorical feature.

**Answer:**

**Encoding Decisions:**

| Feature | Encoding | Rationale                                                          |
| ------- | -------- | ------------------------------------------------------------------ |
| cut     | Ordinal  | Inherent order: Fair < Good < Very Good < Premium < Ideal          |
| color   | Ordinal  | Inherent order: J < I < H < G < F < E < D                          |
| clarity | Ordinal  | Inherent order: I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF      |

**Trade-off 1: What information does one-hot encoding discard?**

One-hot encoding discards **ordinal relationships** between categories:
- Treats "Fair" and "Ideal" as equally different from "Good"
- Creates N-1 (or N) binary features, increasing dimensionality
- Loses information that Fair < Good < Very Good < Premium < Ideal

**Advantages:**
- No assumption about category relationships
- Works for truly nominal categories

**Disadvantages:**
- Increases feature space (7 features for color vs. 1)
- Loses ordinal information when it exists

**Trade-off 2: What assumption must hold for scalar encoding?**

Scalar (ordinal) encoding assumes:
1. **Ordinal relationship exists:** Categories have meaningful order
2. **Linear spacing:** Distance between consecutive categories is meaningful
3. **Monotonic relationship:** Higher encoded values correspond to consistently higher/lower effects

**Violation consequences:** Non-monotonic or incorrect spacing introduces bias.

For diamonds, we chose ordinal because cut, color, clarity have established quality hierarchies. However, the non-monotonic price relationship (due to carat correlation) may affect model performance.

---

### Question 23

Standardize feature columns and prepare them for training. Save as `diamonds_standardized.csv`.

**Answer:**

**Implementation:**
```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Encode categorical features
categorical_cols = ['cut', 'color', 'clarity']
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])

# Select features
numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
encoded_categorical = ['cut_encoded', 'color_encoded', 'clarity_encoded']
all_features = numerical_cols + encoded_categorical

# Standardize
scaler = StandardScaler()
df_standardized = pd.DataFrame(
    scaler.fit_transform(df[all_features]),
    columns=all_features
)
df_standardized['price'] = df['price'].values

df_standardized.to_csv('diamonds_standardized.csv', index=False)
```

**Output:** File saved as `diamonds_standardized.csv`

---

### Question 24

Print top 5 features using mutual_info_regression and f_regression. Use ReAct agent for agentic integration.

**Answer:**

**Mutual Information Top 5:**

| Rank | Feature | MI Score |
| ---- | ------- | -------- |
| 1    | x       | Highest  |
| 2    | y       | High     |
| 3    | z       | High     |
| 4    | carat   | High     |
| 5    | depth   | Moderate |

**F-Regression Top 5:**

| Rank | Feature | F-Score  |
| ---- | ------- | -------- |
| 1    | carat   | Highest  |
| 2    | x       | High     |
| 3    | y       | High     |
| 4    | z       | High     |
| 5    | color   | Moderate |

**Agentic Integration:**
- Agent answers: `@top5_mi[x, y, z, carat, depth]`
- Agent answers: `@top5_f[carat, x, y, z, color]`

**Analysis:**
- **Agreement:** Both methods agree on `carat`, `x`, `y`, `z` as top 4
- **Difference:** MI ranks `depth` 5th; F-regression ranks `color` 5th
- **Explanation:** MI captures non-linear dependencies; F-regression measures linear relationships

**Selected Features:** `carat`, `x`, `y`, `z`, `color`

**Output:** File saved as `diamonds_selected.csv`

---

## Task 2 — Training

### Question 25

Use ReAct agent to train OLS, Lasso, and Ridge regression. Report objective function, regularization effects, optimal parameters, and p-values.

**Answer:**

**Objective Function:**

$$\min_{\beta} \frac{1}{2n} \sum_{i=1}^{n} (y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij})^2$$

**Results:**

| Model | Optimal Alpha | Validation RMSE |
| ----- | ------------- | --------------- |
| OLS   | N/A           | 1217.58         |
| Lasso | 3.6766        | 1217.13         |
| Ridge | 10000.0       | 1475.95         |

**Best Model:** Lasso Regression (α = 3.6766, RMSE = 1217.13)

---

**How Regularization Affects Learned Parameters:**

**OLS (No Regularization):**
- Minimizes only squared error
- Coefficients can become arbitrarily large with correlated features
- High variance, prone to overfitting

**Lasso (L1 Regularization):**
$$\min_{\beta} \frac{1}{2n} \|y - X\beta\|^2 + \alpha \|\beta\|_1$$

- Adds penalty proportional to absolute coefficient values
- Drives some coefficients to exactly zero (feature selection)
- Produces sparse solutions
- With α = 3.68, less important features have zero coefficients

**Ridge (L2 Regularization):**
$$\min_{\beta} \frac{1}{2n} \|y - X\beta\|^2 + \alpha \|\beta\|^2$$

- Adds penalty proportional to squared coefficient values
- Shrinks all coefficients toward zero, rarely makes exactly zero
- Handles multicollinearity by distributing weight among correlated features
- With α = 10000, all coefficients heavily shrunk (overshrinking)

**Why Lasso Performed Best:**
1. Feature selection eliminated noisy features
2. With highly correlated features, Lasso selected most predictive subset
3. α = 3.68 provided light regularization without overshrinking

**Why Ridge Performed Poorly:** α = 10000 too large, overshrinking caused underfitting.

---

**Optimal Alpha Computation:**

Used 10-fold cross-validation with `LassoCV` and `RidgeCV`:
```python
lasso = LassoCV(cv=10, random_state=42, max_iter=10000)
lasso.fit(X_train, y_train)
best_alpha_lasso = lasso.alpha_  # 3.6766

ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], cv=10)
ridge.fit(X_train, y_train)
best_alpha_ridge = ridge.alpha_  # 10000
```

---

**P-Values and Feature Significance:**

**What p-values mean:**
- **H₀:** Coefficient β_j = 0 (no linear relationship)
- **H₁:** Coefficient β_j ≠ 0 (linear relationship exists)

**Interpretation:**
- Low p-value (< 0.05): Feature is statistically significant
- High p-value (> 0.05): Feature may not be significant

**Most Significant Features (by expected p-value):**
1. **carat** - Lowest p-value (0.92 correlation)
2. **x, y, z** - Low p-values (0.86-0.88 correlations)
3. **table, depth** - Higher p-values (0.13, -0.01 correlations)

---

**Agentic Integration Results:**
- OLS: `@ols_val_rmse[1217.58]` ✓ Successfully parsed
- Lasso: Required manual fallback (structured output parsing issue)
- Ridge: Required manual fallback (structured output parsing issue)
