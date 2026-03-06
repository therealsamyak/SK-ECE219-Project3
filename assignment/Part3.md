# Part C: Regression Analysis

## 1 Introduction

Regression analysis is a statistical procedure for estimating the relationship between a target variable and a set of features that jointly inform about the target. In this sub-project, we explore specific-to-regression feature engineering methods and then reuse your ReAct agent to train regression models.

## 2 Datasets

Valentine's day might be over, but we are still interested in building a bot to predict the price and characteristics of diamonds. A synthetic diamonds dataset can be downloaded from this link. This dataset contains information about 150000 round-cut diamonds. There are 14 variables (features) and for each sample, these features specify the various properties of the sample. Below we describe some of these features:

- **carat:** weight of the diamond
- **cut:** quality of the cut
- **clarity:** measured diamond clarity
- **length:** measured length in mm
- **width:** measured width in mm
- **depth:** measured depth in mm
- **depth_percent:** diamond's total height divided by its total width
- **table_percent:** width of top of diamond relative to widest point
- **girdle_min:** refers to the thinnest part of the girdle
- **girdle_max:** refers to the thickest part of the girdle

In addition to these features, there is the target variable: i.e what we would like to predict:

- **price:** price in US dollars

## 3 Training Pipelines

In this section, we describe the setup you need to follow. Follow these steps to process the datasets.

### 3.1 Data Inspection

Before training an algorithm, it's always essential to inspect the data. This provides intuition about the quality and quantity of the data and suggests ideas to extract features for downstream ML applications.

**QUESTION 21:** Perform an exploratory data analysis on the provided Diamonds dataset. Report the following:

- **Correlation Analysis:** Plot a heatmap of the Pearson correlation matrix. Report which features have the highest absolute correlation with the target variable (price). Briefly describe what the correlation patterns suggest.
- **Distribution Analysis:** Plot the histogram of numerical features. Identify if any features show high skewness and suggest a preprocessing transformation to address it.
- **Categorical Analysis:** Construct box plots of categorical features versus the target variable. Describe any significant trends (e.g., how cut or color affects the price range).

### 3.2 Handling Categorical Features

A categorical feature is a feature that can take on one of a limited number of possible values. If one dataset contains categorical features, a preprocessing step needs to be carried to convert categorical variables into numbers and thus prepared for training.

One method for numerical encoding of categorical features is to assign a scalar. For instance, if we have a "Quality" feature with values {Poor, Fair, Typical, Good, Excellent} we might replace them with numbers 1 through 5. If there is no numerical meaning behind categorical features (e.g. {Cat, Dog}) one has to perform "one-hot encoding" instead.

**QUESTION 22:** Explain the following trade-off questions.

- Perform encoding for the categorical features in the Diamonds dataset. Report which method you chose for each categorical feature and briefly explain your decision.
- Explain the following trade-offs:
  - What information does one-hot encoding discard?
  - What assumption should hold strongly if we perform the scalar encoding instead?

#### 3.2.1 Standardization

Standardization of datasets is a common requirement for many machine learning estimators; they might behave badly if the individual features do not more-or-less look like standard normally distributed data: Gaussian with zero mean and unit variance. If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.

**QUESTION 23:** Standardize feature columns and prepare them for training. Save your standardized version of the dataset as `diamonds_standardized.csv`.

#### 3.2.2 Feature Selection

- `sklearn.feature_selection.mutual_info_regression` function returns estimated mutual information between each feature and the label. Mutual information (MI) between two random variables is a non-negative value which measures the dependency between the variables. It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.
- `sklearn.feature_selection.f_regression` function provides F scores, which is a way of comparing the significance of the improvement of a model, with respect to the addition of new variables.

You may use these functions to select features that yield better regression results (especially in the classical models).

**QUESTION 24:** Print the top 5 features using each method (`mutual_info_regression` and `f_regression`).

- **Agentic Integration:** For this step, load `diamonds-questions.jsonl` and `diamonds-labels.jsonl` (question id 0 and 1) and use your ReAct agent from Part 2 to automatically identify and print the top features. If the agent gets stuck, you may manually write the code to compute and print them.

From this point on, you are free to use any combination of features, as long as the performance on the regression model is on par (or slightly worse) than the Neural Network model.

Save your selected feature new csv as `diamonds_selected.csv`.

### 3.3 Training

Once the data is prepared, we would like to train multiple algorithms and compare their performance using average RMSE from 10-fold cross-validation (please refer to part 3.4).

### 3.4 Evaluation

Perform 10-fold cross-validation and measure average RMSE errors for training and validation sets.

#### Task 5.1 Linear Regression

**QUESTION 25:** Agentic Integration: For this step, load `diamonds-questions.jsonl` and `diamonds-labels.jsonl` (question ids 2, 3, and 4) and use your ReAct agent from Part 2 to automatically train the models and extract the necessary metrics. If the agent gets stuck, you may manually write the code to complete the training.

**Important:** List out the Python code generated by your agent for these tasks. Review the code to ensure it makes sense and correctly implements the requested regression models. Do not blindly trust the agent's output; verify its logic before proceeding.

What is the objective function? Train three models: (a) ordinary least squares (linear regression without regularization), (b) Lasso and (c) Ridge regression, and answer the following questions.

- Explain how each regularization scheme affects the learned parameter set.
- Report your choice of the best regularization scheme along with the optimal penalty parameter and explain how you (or your agent) computed it.
- Some linear regression packages return p-values for different features. What is the meaning of these p-values and how can you infer the most significant features? A qualitative reasoning is sufficient.
