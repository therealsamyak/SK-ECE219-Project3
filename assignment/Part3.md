## Part C: Regression Analysis and Agent-Assisted Feature Engineering

### Introduction

In Parts A and B, you developed a fine-tuned reasoning model and a ReAct-style data analysis agent. Part C brings these threads together in the context of a classical machine learning task: predicting housing prices using regression models on the Ames Housing dataset.

The goal here is not to build the most accurate model possible, but to practice a principled regression workflow — understanding your data, engineering meaningful features, fitting and diagnosing models, and interpreting results. You will also integrate your agent from Part B as an optional but encouraged tool for exploratory data analysis and feature engineering ideation.

---

### Background: Regression Modeling

Linear regression models the relationship between a response variable y and a set of predictors X as:

$$y = X\beta + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

The ordinary least squares (OLS) estimator minimizes the residual sum of squares:

$$\hat{\beta} = \arg\min_\beta \|y - X\beta\|^2 = (X^TX)^{-1}X^Ty$$

In practice, raw predictors often violate assumptions (linearity, homoscedasticity, normality of residuals) or are collinear, which degrades model performance and interpretability. Feature engineering and regularization are standard remedies.

---

### Task 9 — Exploratory Data Analysis

Before building any model, understand the data.

**QUESTION 20: (10 points)** Perform an exploratory data analysis of the Ames Housing dataset. Your analysis should include:

- Distribution of the target variable `SalePrice` (histogram, skewness).
- Correlation heatmap of the top numerical features with `SalePrice`.
- Identification of missing values: which columns have the most missing data, and what do you plan to do about them?

You are encouraged to use your ReAct agent from Part B to assist with parts of this analysis.

**QUESTION 21: (10 points)** Based on your EDA, select 8–12 numerical features that you believe are most predictive of `SalePrice`. Justify your selection using correlation values and domain reasoning (e.g., why would `GrLivArea` be a strong predictor?).

---

### Task 10 — Baseline Regression Model

**QUESTION 22: (20 points)** Fit an OLS linear regression model on your selected features (use an 80/20 train/test split). Report:

- Train and test R²,
- Train and test RMSE,
- A coefficient table showing the sign and magnitude of each feature's coefficient.

Interpret at least 3 coefficients in plain language (e.g., "holding all else equal, one additional square foot of above-ground living area is associated with a $X increase in sale price").

**QUESTION 23: (15 points)** Diagnose your baseline model using residual analysis:

- Plot residuals vs. fitted values. Do you see any patterns (heteroscedasticity, nonlinearity)?
- Plot a Q-Q plot of the residuals. Are they approximately normally distributed?
- Identify any high-leverage or high-influence points (e.g., using Cook's distance).

Based on your diagnostics, what are the two biggest modeling problems you observe?

---

### Task 11 — Feature Engineering

**QUESTION 24: (20 points)** Apply at least three of the following feature engineering techniques and re-fit your regression model after each:

- **Log-transform the target:** replace `SalePrice` with `log(SalePrice)` to reduce skewness and stabilize variance.
- **Log-transform skewed predictors:** apply log transforms to right-skewed numerical features (e.g., `LotArea`, `GrLivArea`).
- **Polynomial features:** add squared terms for features with nonlinear relationships with `SalePrice`.
- **Interaction terms:** create interaction features between pairs of variables that have meaningful joint effects (e.g., `OverallQual × GrLivArea`).
- **Encoding categorical variables:** one-hot encode key categorical features (e.g., `Neighborhood`, `BldgType`) and add them to the model.

For each technique, report the new train/test R² and RMSE, and describe whether and why it helped.

**QUESTION 25: (10 points)** After applying your feature engineering pipeline, re-run your residual diagnostics from Question 23. Has heteroscedasticity improved? Are residuals more normally distributed? Summarize what changed and what remains problematic.

---

### Task 12 — Regularization

When the feature space grows (especially after adding polynomial and interaction terms), OLS can overfit. Regularization constrains the coefficient magnitudes to reduce variance at the cost of some bias.

**Ridge regression** adds an L2 penalty:

$$\hat{\beta}_{\text{ridge}} = \arg\min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|^2$$

**Lasso regression** adds an L1 penalty:

$$\hat{\beta}_{\text{lasso}} = \arg\min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_1$$

The key difference: Lasso produces sparse solutions (some coefficients are exactly zero), making it useful for feature selection.

**QUESTION 26: (20 points)** Fit both Ridge and Lasso regression models on your engineered feature set. For each:

- Use cross-validation to select the regularization strength λ.
- Report train and test R² and RMSE.
- For Lasso: report how many features were zeroed out and list the top 10 features by absolute coefficient magnitude.

Compare Ridge, Lasso, and your best OLS model in a summary table.

**QUESTION 27: (10 points)** Interpret the regularization results:

- Did regularization improve test performance? By how much?
- Which model (Ridge or Lasso) performed better, and why might that be the case for this dataset?
- What does the sparsity pattern from Lasso tell you about feature importance?

---

### Task 13 — Agent-Assisted Feature Engineering (Integration Task)

**QUESTION 28: (20 points)** Use your ReAct agent from Part B to assist with at least one of the following:

1. **Automated feature suggestion:** Ask the agent to identify which features have the strongest nonlinear relationships with `SalePrice` (e.g., by computing rank correlations or binned mean plots), and use its output to motivate one new feature you add to your model.
2. **Outlier investigation:** Ask the agent to identify statistical outliers in your residuals (e.g., homes where your model is off by more than 2 standard deviations), then manually inspect those homes and describe what makes them unusual.
3. **Interaction discovery:** Ask the agent to test whether specific interaction terms are statistically significant (e.g., using a partial F-test or comparing model R² with and without the interaction), and report what it finds.

For this question, include:

- The question(s) you posed to the agent,
- The full agent trace (Thought/Action/Observation steps),
- How you used the agent's output to improve or inform your regression model,
- The resulting change in model performance (if any).

---

### Task 14 — Final Model and Reflection

**QUESTION 29: (15 points)** Present your final regression model. This should be your best model after all feature engineering and regularization. Report:

- Final train and test R² and RMSE,
- The complete list of features used,
- A coefficient plot or table showing the most important predictors,
- A brief narrative explaining why this model is better than your baseline.

**QUESTION 30: (10 points)** Final reflection: Write a short paragraph (6–10 sentences) addressing the following:

- What was the single most impactful modeling decision you made (feature engineering step, regularization choice, target transformation)?
- What was the most surprising finding from your analysis?
- If you had more time, what would you try next to further improve the model?
- How did integrating the ReAct agent into your workflow change (or not change) how you approached the analysis?