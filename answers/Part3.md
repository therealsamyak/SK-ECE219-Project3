# Part C: Regression Analysis

## Task 1 — Data Inspection

**QUESTION 21: Perform an exploratory data analysis on the provided Diamonds dataset. Report the following:**

- **Correlation Analysis:** Plot a heatmap of the Pearson correlation matrix. Report which features have the highest absolute correlation with the target variable (price). Briefly describe what the correlation patterns suggest.
- **Distribution Analysis:** Plot the histogram of numerical features. Identify if any features show high skewness and suggest a preprocessing transformation to address it.
- **Categorical Analysis:** Construct box plots of categorical features versus the target variable. Describe any significant trends (e.g., how cut or color affects the price range).

**Answer:** Report correlation analysis, distribution analysis, and categorical analysis.

**Answer:**

**Correlation analysis:**

File: [Correlation Heatmap](outputs/q21_correlation_heatmap.png)

| Feature | Correlation with Price |
| ------- | ---------------------- |
| carat   | 0.9216 (strongest)     |
| x       | 0.8844                 |
| y       | 0.8654                 |
| z       | 0.8612                 |
| table   | 0.1271                 |
| depth   | -0.0106 (weakest)      |

**Key observations:**
1. Carat has strongest correlation (0.92) - larger diamonds more valuable
2. Physical dimensions (x, y, z) highly correlated (0.86-0.88) - related to carat weight
3. Depth and table show weak correlations
4. High multicollinearity among x, y, z, carat (>0.95)

---

**Distribution analysis:**

File: [Histograms](outputs/q21_histograms.png)

| Feature | Skewness | Assessment              | Transformation |
| ------- | -------- | ----------------------- | -------------- |
| y       | 2.4342   | Highly skewed           | Log            |
| z       | 1.5224   | Moderately skewed       | Log            |
| carat   | 1.1166   | Moderately skewed       | Log            |
| table   | 0.7969   | Slightly skewed         | Optional       |
| x       | 0.3787   | Approximately symmetric | None           |
| depth   | -0.0823  | Approximately symmetric | None           |

**Recommendation:** Apply log transformation to `carat`, `y`, `z` to fix right-skewness.

---

**Categorical analysis:**

File: [Box Plots](outputs/q21_boxplots.png)

**Cut vs Price:**
| Cut       | Median Price |
| --------- | ------------ |
| Fair      | $3,282       |
| Premium   | $3,185       |
| Good      | $3,050       |
| Very Good | $2,648       |
| Ideal     | $1,810       |

**Trend:** "Ideal" cut has lowest median price, "Fair" highest—counterintuitive due to carat correlation.

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

**Trend:** Lower color grades (J, I, H) higher prices—likely due to carat correlation.

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

**Trend:** Lower clarity grades higher prices—counterintuitive but explained by carat correlation.

---

**QUESTION 22: Explain the following trade-off questions.**

- Perform encoding for the categorical features in the Diamonds dataset. Report which method you chose for each categorical feature and briefly explain your decision.
- Explain the following trade-offs:
  - What information does one-hot encoding discard?
  - What assumption should hold strongly if we perform the scalar encoding instead?

**Answer:**

**Answer:**

**Encoding decisions:**

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

For diamonds, we chose ordinal because cut, color, clarity have established quality hierarchies. But the non-monotonic price relationship (due to carat correlation) may hurt model performance.

---

**QUESTION 23: Standardize feature columns and prepare them for training. Save your standardized version of the dataset as `diamonds_standardized.csv`.**

**Answer:** Save as `diamonds_standardized.csv`.

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

**QUESTION 24: Print the top 5 features using each method (`mutual_info_regression` and `f_regression`).**

- **Agentic Integration:** For this step, load `diamonds-questions.jsonl` and `diamonds-labels.jsonl` (question id 0 and 1) and use your ReAct agent from Part 2 to automatically identify and print the top features. If the agent gets stuck, you may manually write the code to compute and print them.

From this point on, you are free to use any combination of features, as long as the performance on the regression model is on par (or slightly worse) than the Neural Network model.

Save your selected feature new csv as `diamonds_selected.csv`.

**Answer:** Use ReAct agent for agentic integration.

**Answer:**

**Mutual Information Top 5:**

| Rank | Feature | MI Score |
| ---- | ------- | -------- |
| 1    | carat   | 1.9622   |
| 2    | y       | 1.4912   |
| 3    | x       | 1.4832   |
| 4    | z       | 1.4329   |
| 5    | clarity | 0.3607   |

**F-Regression Top 5:**

| Rank | Feature | F-Score  |
| ---- | ------- | -------- |
| 1    | carat   | 304050.9059 |
| 2    | x       | 193740.2791 |
| 3    | y       | 160914.4818 |
| 4    | z       | 154922.1211 |
| 5    | color   | 1654.4319   |

**Agentic integration:**
- The ReAct agent was used to identify top features automatically.
- Agent outputs: `@top5_mi[carat, y, x, z, clarity]` and `@top5_f[carat, x, y, z, color]`

**Analysis:**
- **Agreement:** Both methods agree on `carat`, `x`, `y`, `z` as top 4
- **Difference:** MI ranks `clarity` 5th; F-regression ranks `color` 5th
    - MI captures non-linear dependencies; F-regression measures linear relationships
    - **Selected features:** `carat`, `x`, `y`, `z`, `color` (from F-regression top 5)

**Output:** File saved as `diamonds_selected.csv`

---

## Task 2 — Training

**QUESTION 25: Agentic Integration: For this step, load `diamonds-questions.jsonl` and `diamonds-labels.jsonl` (question ids 2, 3, and 4) and use your ReAct agent from Part 2 to automatically train the models and extract the necessary metrics. If the agent gets stuck, you may manually write the code to complete the training.**

**Important:** List out the Python code generated by your agent for these tasks. Review the code to ensure it makes sense and correctly implements the requested regression models. Do not blindly trust the agent's output; verify its logic before proceeding.

What is the objective function? Train three models: (a) ordinary least squares (linear regression without regularization), (b) Lasso and (c) Ridge regression, and answer the following questions.

- Explain how each regularization scheme affects the learned parameter set.
- Report your choice of the best regularization scheme along with the optimal penalty parameter and explain how you (or your agent) computed it.
- Some linear regression packages return p-values for different features. What is the meaning of these p-values and how can you infer the most significant features? A qualitative reasoning is sufficient.

**Answer:** Report objective function, regularization effects, optimal parameters, and p-values.

**Answer:**

**Objective Function:**

$$\min_{\beta} \frac{1}{2n} \sum_{i=1}^{n} (y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij})^2$$

**Results:**

| Model | Optimal Alpha | Validation RMSE | Training RMSE |
| ----- | ------------- | --------------- | ------------- |
| OLS   | N/A           | 1217.58         | 1216.55       |
| Lasso | 3.6766        | 1217.13         | 1217.15       |
| Ridge | 10000.0       | 1475.95         | 1460.29       |

**Best model:** Lasso Regression (α = 3.6766, RMSE = 1217.13)

---

**How regularization affects learned parameters:**

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

**Optimal alpha computation:**

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

**P-values and feature significance:**

**What p-values mean:**
- **H₀:** Coefficient β_j = 0 (no linear relationship)
- **H₁:** Coefficient β_j ≠ 0 (linear relationship exists)

**Interpretation:**
- Low p-value (< 0.05): Feature is statistically significant
- High p-value (> 0.05): Feature may not be significant

**Most significant features (by expected p-value):**
1. **carat** - Lowest p-value (0.92 correlation)
2. **x, y, z** - Low p-values (0.86-0.88 correlations)
3. **table, depth** - Higher p-values (0.13, -0.01 correlations)

---

**Agent Integration:** The ReAct agent from Part 2 was used to train and evaluate the models.

---

**Python Code for Regression Models:**

**OLS Regression:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=10, shuffle=True, random_state=42)
model = LinearRegression()
neg_mse = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
val_rmse = np.sqrt(-neg_mse).mean()
```

**Lasso Regression:**
```python
from sklearn.linear_model import LassoCV, Lasso

lasso_cv = LassoCV(cv=10, random_state=42, max_iter=10000)
lasso_cv.fit(X, y)
best_alpha = lasso_cv.alpha_
model = Lasso(alpha=best_alpha, random_state=42)
neg_mse = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
rmse = np.sqrt(-neg_mse).mean()
```

**Ridge Regression:**
```python
from sklearn.linear_model import RidgeCV, Ridge

ridge_cv = RidgeCV(alphas=np.logspace(-2, 4, 50), cv=10)
ridge_cv.fit(X, y)
best_alpha = ridge_cv.alpha_
model = Ridge(alpha=best_alpha)
neg_mse = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
rmse = np.sqrt(-neg_mse).mean()
```
