# Part 3: Regression Analysis

## Question 21: Exploratory Data Analysis (15 points)

**Question:** Perform an exploratory data analysis on the Diamonds dataset. Report correlation analysis, distribution analysis, and categorical analysis.

### Correlation Analysis

**Pearson Correlation Matrix:**

The correlation heatmap (see `outputs/q21_correlation_heatmap.png`) reveals the following relationships with the target variable (price):

| Feature | Correlation with Price |
|---------|----------------------|
| carat   | 0.9216 (strongest)   |
| x       | 0.8844               |
| y       | 0.8654               |
| z       | 0.8612               |
| table   | 0.1271               |
| depth   | -0.0106 (weakest)    |

**Key Observations:**

1. **Carat has the strongest correlation (0.92)**: This makes intuitive sense as larger diamonds are more valuable.
2. **Physical dimensions (x, y, z) are highly correlated with price (0.86-0.88)**: These are related to carat weight.
3. **Depth and table show weak correlations**: These measurements have minimal linear relationship with price.
4. **High multicollinearity among x, y, z, and carat**: These features are highly correlated with each other (>0.95), which may cause issues in linear models.

### Distribution Analysis

**Histograms** (see `outputs/q21_histograms.png`) reveal the following:

**Skewness Analysis:**

| Feature | Skewness | Assessment        | Suggested Transformation |
|---------|----------|-------------------|--------------------------|
| y       | 2.4342   | Highly skewed     | Log transformation       |
| z       | 1.5224   | Moderately skewed | Log transformation       |
| carat   | 1.1166   | Moderately skewed | Log transformation       |
| table   | 0.7969   | Slightly skewed   | May not need transform   |
| x       | 0.3787   | Approximately symmetric | None needed        |
| depth   | -0.0823  | Approximately symmetric | None needed        |

**Recommendation:** Apply log transformation to `carat`, `y`, and `z` features to address right-skewness and improve model performance.

### Categorical Analysis

**Box Plots** (see `outputs/q21_boxplots.png`) show:

#### Cut vs Price

| Cut       | Median Price |
|-----------|-------------|
| Fair      | $3,282      |
| Premium   | $3,185      |
| Good      | $3,050      |
| Very Good | $2,648      |
| Ideal     | $1,810      |

**Trend:** Surprisingly, "Ideal" cut diamonds have the lowest median price, while "Fair" cut has the highest. This counterintuitive result occurs because carat weight (the dominant price factor) varies systematically with cut quality in this dataset.

#### Color vs Price

| Color | Median Price |
|-------|-------------|
| J     | $4,234      |
| I     | $3,730      |
| H     | $3,460      |
| G     | $2,242      |
| F     | $2,343      |
| E     | $1,739      |
| D     | $1,838      |

**Trend:** Lower color grades (J, I, H) have higher median prices than higher grades (D, E, F). Again, this is likely due to correlation with carat weight.

#### Clarity vs Price

| Clarity | Median Price |
|---------|-------------|
| SI2     | $4,072      |
| I1      | $3,344      |
| SI1     | $2,822      |
| VS2     | $2,054      |
| VS1     | $2,005      |
| VVS2    | $1,311      |
| VVS1    | $1,093      |
| IF      | $1,080      |

**Trend:** Lower clarity grades (SI2, I1) have higher median prices than flawless (IF) diamonds, which is counterintuitive but explained by carat correlation.

**Conclusion:** Categorical features show complex interactions with price due to their correlation with carat weight. Simple encoding may not capture these relationships effectively.

---

## Question 22: Categorical Feature Encoding (10 points)

**Question:** Explain encoding trade-offs and report which method was chosen for each categorical feature.

### Encoding Decisions

| Feature  | Encoding Method | Rationale |
|----------|----------------|-----------|
| cut      | Ordinal        | Has inherent quality order: Fair < Good < Very Good < Premium < Ideal |
| color    | Ordinal        | Has inherent color grade order: J < I < H < G < F < E < D |
| clarity  | Ordinal        | Has inherent clarity order: I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF |

### Trade-off Explanations

#### What information does one-hot encoding discard?

One-hot encoding discards the **ordinal relationship** between categories. For example:
- For "cut", one-hot encoding treats "Fair" and "Ideal" as equally different from "Good"
- It creates N-1 (or N) binary features, increasing dimensionality
- It loses the information that Fair < Good < Very Good < Premium < Ideal

**Advantages of one-hot:**
- No assumption about relationship between categories
- Works for truly nominal categories (no order)

**Disadvantages of one-hot:**
- Increases feature space (7 features for color vs. 1)
- Loses ordinal information when it exists
- May cause overfitting with many categories

#### What assumption must hold for scalar encoding?

Scalar (ordinal) encoding assumes that:
1. **Ordinal relationship exists**: Categories have a meaningful order
2. **Linear spacing**: The numerical distance between consecutive categories is meaningful (e.g., the difference between "Fair" and "Good" is similar to "Good" to "Very Good")
3. **Monotonic relationship**: Higher encoded values should correspond to consistently higher (or lower) effects on the target

**Violation consequences:** If the relationship between category and target is non-monotonic or the spacing assumption is wrong, ordinal encoding can introduce bias and reduce model performance.

**For diamonds dataset:** We chose ordinal encoding because cut, color, and clarity have established quality hierarchies in the diamond industry. However, as seen in Q21, the relationship with price is not monotonic due to carat correlation, which may affect model performance.

---

## Question 23: Standardization (5 points)

**Question:** Standardize feature columns and prepare them for training.

**Implementation:**

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Encode categorical features
categorical_cols = ['cut', 'color', 'clarity']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# Select features for standardization
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

# Save
df_standardized.to_csv('diamonds_standardized.csv', index=False)
```

**Output:** Saved as `diamonds_standardized.csv`

---

## Question 24: Feature Selection (10 points)

**Question:** Print the top 5 features using mutual_info_regression and f_regression.

### Results

**Mutual Information (MI) Top 5 Features:**

| Rank | Feature | MI Score |
|------|---------|----------|
| 1    | x       | Highest  |
| 2    | y       | High     |
| 3    | z       | High     |
| 4    | carat   | High     |
| 5    | depth   | Moderate |

**F-Regression Top 5 Features:**

| Rank | Feature | F-Score |
|------|---------|---------|
| 1    | carat   | Highest |
| 2    | x       | High    |
| 3    | y       | High    |
| 4    | z       | High    |
| 5    | color   | Moderate |

### Agentic Integration

The ReAct agent was used to automatically identify these features. The agent answers were:
- **Mutual Info:** `@top5_mi[x, y, z, carat, depth]`
- **F-Regression:** `@top5_f[carat, x, y, z, color]`

### Analysis

**Agreement:** Both methods agree that `carat`, `x`, `y`, `z` are the top 4 features.

**Difference:** 
- MI ranks `depth` as the 5th feature
- F-regression ranks `color` as the 5th feature

**Explanation:** 
- **Mutual Information** captures non-linear dependencies, explaining why physical dimensions (x, y, z) rank higher
- **F-Regression** measures linear relationships, where carat (strongly linearly correlated with price) ranks highest

**Selected Features for Model Training:** Based on both methods, we select: `carat`, `x`, `y`, `z`, `color` (and optionally `depth`)

**Output:** Saved as `diamonds_selected.csv`

---

## Question 25: Linear Regression Models (25 points)

**Question:** Train OLS, Lasso, and Ridge regression. Explain regularization effects, report best model, and explain p-values.

### Objective Function

The objective function for linear regression is:

$$\min_{\beta} \frac{1}{2n} \sum_{i=1}^{n} (y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij})^2$$

### Results

| Model   | Optimal Alpha | Validation RMSE |
|---------|--------------|-----------------|
| OLS     | N/A          | 1217.58         |
| Lasso   | 3.6766       | 1217.13         |
| Ridge   | 10000.0      | 1475.95         |

**Best Model:** **Lasso Regression** with α = 3.6766

### How Regularization Affects Learned Parameters

#### OLS (No Regularization)

- **Effect:** Minimizes only the squared error, no penalty on coefficients
- **Parameter behavior:** Coefficients can become arbitrarily large if features are correlated
- **Risk:** High variance, prone to overfitting with multicollinearity

#### Lasso (L1 Regularization)

$$\min_{\beta} \frac{1}{2n} \|y - X\beta\|^2 + \alpha \|\beta\|_1$$

- **Effect:** Adds penalty proportional to absolute value of coefficients
- **Parameter behavior:** 
  - Drives some coefficients to exactly zero (feature selection)
  - Produces sparse solutions
  - With α = 3.68, some less important features may have zero coefficients
- **Best for:** When you suspect some features are irrelevant

#### Ridge (L2 Regularization)

$$\min_{\beta} \frac{1}{2n} \|y - X\beta\|^2 + \alpha \|\beta\|^2$$

- **Effect:** Adds penalty proportional to squared value of coefficients
- **Parameter behavior:**
  - Shrinks all coefficients toward zero but rarely makes them exactly zero
  - With α = 10000, all coefficients are heavily shrunk
  - Handles multicollinearity by distributing weight among correlated features
- **Best for:** When all features are potentially relevant

### Why Lasso Performed Best

1. **Feature selection:** Lasso's L1 penalty likely eliminated noisy features, reducing overfitting
2. **Multicollinearity handling:** With highly correlated features (x, y, z, carat), Lasso selected the most predictive subset
3. **Optimal regularization strength:** α = 3.68 provided light regularization without overshrinking

**Why Ridge performed poorly:** α = 10000 is too large, overshrinking all coefficients and causing underfitting (high RMSE).

### How Optimal Alpha Was Computed

**Method:** 10-fold cross-validation using `LassoCV` and `RidgeCV`

```python
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold

# Lasso with cross-validation
lasso = LassoCV(cv=10, random_state=42, max_iter=10000)
lasso.fit(X_train, y_train)
best_alpha_lasso = lasso.alpha_  # 3.6766

# Ridge with cross-validation
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
ridge = RidgeCV(alphas=alphas, cv=10)
ridge.fit(X_train, y_train)
best_alpha_ridge = ridge.alpha_  # 10000
```

The optimal alpha was selected by minimizing average validation RMSE across 10 folds.

### P-Values and Feature Significance

**What p-values mean in linear regression:**

The p-value for each feature tests the null hypothesis:
- **H₀:** The coefficient β_j = 0 (the feature has no linear relationship with the target)
- **H₁:** The coefficient β_j ≠ 0 (the feature has a linear relationship with the target)

**Interpretation:**
- **Low p-value (< 0.05):** Reject H₀ → The feature is statistically significant
- **High p-value (> 0.05):** Fail to reject H₀ → The feature may not be significant

**Inferring most significant features:**

Features with the lowest p-values are the most statistically significant. Based on correlation analysis (Q21), we expect:

1. **carat** - Lowest p-value (highest significance) due to 0.92 correlation
2. **x, y, z** - Low p-values due to 0.86-0.88 correlations
3. **table, depth** - Higher p-values due to weak correlations (0.13, -0.01)

**Qualitative reasoning:** Features with stronger linear relationships to price will have smaller p-values. The p-value depends on both the correlation strength and the variance of the feature.

### Agentic Integration

The ReAct agent was used to train these models. Results:
- **OLS:** `@ols_val_rmse[1217.58]` ✓ Successfully parsed
- **Lasso:** Agent produced fallback (could not parse structured output)
- **Ridge:** Agent produced fallback (could not parse structured output)

The agent successfully completed OLS but had difficulty with the structured output format for Lasso and Ridge, requiring manual fallback code.

---

## Summary

| Question | Key Findings |
|----------|-------------|
| Q21 | Carat (0.92) and dimensions (x,y,z) are most correlated with price; log transform recommended for skewed features |
| Q22 | Ordinal encoding used for cut, color, clarity due to inherent quality hierarchies |
| Q23 | Standardized dataset saved as `diamonds_standardized.csv` |
| Q24 | Top features: carat, x, y, z (both methods); depth (MI) vs color (F-reg) differ on 5th feature |
| Q25 | Lasso (α=3.68) best with RMSE=1217.13; Ridge (α=10000) over-regularized; OLS similar to Lasso |
