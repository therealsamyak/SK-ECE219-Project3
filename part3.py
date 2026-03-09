"""
Part 3: Diamond Dataset Regression Analysis
Q21: Exploratory Data Analysis (Correlation, Distribution, Categorical)
Q23: Feature Standardization
Q24: Feature Selection (Mutual Information, F-regression) with ReAct Agent
Q25: Regression Models (OLS, Lasso, Ridge) with ReAct Agent
"""

import json
import os
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from part2 import ReActAgent, load_jsonl

# Constants
OUTPUT_DIR = "outputs"
DIAMONDS_PATH = "datasets/share_data/da-dev-tables/diamonds.csv"
TABLES_DIR = "datasets/share_data/da-dev-tables"

# Categorical ordering for ordinal encoding
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]  # J=worst, D=best
CLARITY_ORDER = [
    "I1",
    "SI2",
    "SI1",
    "VS2",
    "VS1",
    "VVS2",
    "VVS1",
    "IF",
]  # I1=worst, IF=best

# Numerical and categorical columns
NUMERICAL_COLS = ["carat", "depth", "table", "price", "x", "y", "z"]
CATEGORICAL_COLS = ["cut", "color", "clarity"]

# Q24/Q25 paths
QUESTIONS_PATH = "datasets/share_data/diamonds-questions.jsonl"
STANDARDIZED_PATH = "datasets/share_data/da-dev-tables/diamonds_standardized.csv"
SELECTED_PATH = "datasets/share_data/da-dev-tables/diamonds_selected.csv"


def save_json(data, filename: str):
    """Save data to outputs/{filename}, pretty-printed.

    Args:
        data: Data to save (dict, list, etc.).
        filename: Name of the file to save (relative to OUTPUT_DIR).
    """
    filepath = Path(OUTPUT_DIR) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {filepath}")


def save_plot(fig, filename: str):
    """Save matplotlib figure to outputs/{filename} and close it.

    Args:
        fig: matplotlib figure object.
        filename: Name of the file to save (relative to OUTPUT_DIR).
    """
    filepath = Path(OUTPUT_DIR) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


def run_q21_correlation():
    """Q21 Correlation: Compute Pearson correlation matrix and extract top correlations with price."""
    print("=== Q21: Correlation Analysis ===")

    # Load data
    df = pd.read_csv(DIAMONDS_PATH)

    # Select numerical features for correlation
    numerical_df = df[NUMERICAL_COLS]

    # Compute Pearson correlation matrix
    corr_matrix = numerical_df.corr(method="pearson")

    # Extract correlations with price (excluding price itself)
    price_correlations = (
        corr_matrix["price"].drop("price").abs().sort_values(ascending=False)
    )

    # Get top correlations
    top_correlations = {}
    for feature, corr_value in price_correlations.items():
        actual_corr = corr_matrix.loc[feature, "price"]
        top_correlations[feature] = round(actual_corr, 4)

    # Save correlation results as JSON
    result = {
        "correlation_matrix": corr_matrix.round(4).to_dict(),
        "top_correlations": top_correlations,
    }
    save_json(result, "q21_correlation.json")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix - Numerical Features")
    save_plot(fig, "q21_correlation_heatmap.png")


def run_q21_distribution():
    """Q21 Distribution: Compute skewness and identify highly skewed features."""
    print("=== Q21: Distribution Analysis ===")

    # Load data
    df = pd.read_csv(DIAMONDS_PATH)

    # Select numerical features (excluding price for transformation suggestions)
    numerical_features = ["carat", "depth", "table", "x", "y", "z"]

    # Compute skewness
    skewness = {}
    skewed_features = {}

    for col in numerical_features:
        skew_val = df[col].skew()
        skewness[col] = round(skew_val, 4)

        # Identify highly skewed features (|skewness| > 1)
        if abs(skew_val) > 1:
            transformation = "log" if skew_val > 1 else "sqrt"
            skewed_features[col] = {
                "skewness": round(skew_val, 4),
                "suggested_transformation": transformation,
            }

    # Save distribution results as JSON
    result = {"skewness": skewness, "skewed_features": skewed_features}
    save_json(result, "q21_distribution.json")

    # Create histogram grid for numerical features
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, col in enumerate(numerical_features):
        axes[idx].hist(df[col], bins=50, edgecolor="black", alpha=0.7)
        axes[idx].set_title(f"{col}\n(skewness: {skewness[col]:.2f})")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Frequency")

    plt.suptitle("Distribution of Numerical Features", fontsize=14)
    plt.tight_layout()
    save_plot(fig, "q21_histograms.png")


def run_q21_categorical():
    """Q21 Categorical: Generate box plots and calculate median prices per category."""
    print("=== Q21: Categorical Analysis ===")

    # Load data
    df = pd.read_csv(DIAMONDS_PATH)

    orders = {"cut": CUT_ORDER, "color": COLOR_ORDER, "clarity": CLARITY_ORDER}
    categorical_trends = {}

    for col in CATEGORICAL_COLS:
        median_prices = df.groupby(col)["price"].median().to_dict()
        median_prices = {k: int(v) for k, v in median_prices.items()}
        ordered_median_prices = {
            cat: median_prices[cat] for cat in orders[col] if cat in median_prices
        }

        categorical_trends[col] = {
            "median_prices": ordered_median_prices,
            "categories": orders[col],
        }

    # Save categorical results as JSON
    result = {"categorical_trends": categorical_trends}
    save_json(result, "q21_categorical.json")

    # Create box plots for categorical features vs price
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, col in enumerate(CATEGORICAL_COLS):
        sns.boxplot(data=df, x=col, y="price", order=orders[col], ax=axes[idx])
        axes[idx].set_title(f"Price by {col.capitalize()}")
        axes[idx].set_xlabel(col.capitalize())
        axes[idx].set_ylabel("Price ($)")
        axes[idx].tick_params(axis="x", rotation=45)

    plt.suptitle("Price Distribution by Categorical Features", fontsize=14)
    plt.tight_layout()
    save_plot(fig, "q21_boxplots.png")


def run_q23_standardization():
    """Q23 Standardization: Encode categoricals and apply StandardScaler."""
    print("=== Q23: Feature Standardization ===")

    # Load data
    df = pd.read_csv(DIAMONDS_PATH)

    # Create a copy for standardization
    df_std = df.copy()

    # Ordinal encoding for categorical features
    categories_order = [CUT_ORDER, COLOR_ORDER, CLARITY_ORDER]
    encoder = OrdinalEncoder(categories=categories_order)
    df_std[CATEGORICAL_COLS] = encoder.fit_transform(df_std[CATEGORICAL_COLS])

    # Separate features and target (do NOT standardize price)
    features_to_scale = [col for col in df_std.columns if col != "price"]

    # Apply StandardScaler to all features except price
    scaler = StandardScaler()
    df_std[features_to_scale] = scaler.fit_transform(df_std[features_to_scale])

    # Save standardized dataset
    output_path = Path(TABLES_DIR) / "diamonds_standardized.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_std.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    # Verify standardization (print statistics)
    print("\nStandardization verification (should be mean≈0, std≈1):")
    for col in features_to_scale:
        mean_val = df_std[col].mean()
        std_val = df_std[col].std()
        print(f"  {col}: mean={mean_val:.6f}, std={std_val:.6f}")


def main():
    """Run all Part 3 analyses: Q21, Q23, Q24, Q25."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Q21: Exploratory Data Analysis
    run_q21_correlation()
    run_q21_distribution()
    run_q21_categorical()

    # Q23: Feature Standardization
    run_q23_standardization()

    # Q24: Feature Selection via ReAct Agent
    agent = run_q24_feature_selection()

    # Q25: Regression Models via ReAct Agent
    run_q25_regression(agent)

    print("\n=== All tasks completed ===")


def run_q24_feature_selection():
    """Q24: Feature selection using ReAct agent (mutual info + f-regression)."""
    print("=== Q24: Feature Selection ===")

    questions = load_jsonl(QUESTIONS_PATH)
    df = pd.read_csv(STANDARDIZED_PATH)

    X = df.drop(columns=["price"])
    y = df["price"]
    feature_names = X.columns.tolist()

    agent = ReActAgent()
    results = {"agent_answers": {}}

    # Question ID 0: Mutual Information
    q0 = questions[0]
    try:
        print(f"Running agent for Q24 ID 0: {q0['question'][:80]}...")
        answer, history = agent.run(
            question=q0["question"],
            constraints=q0["constraints"],
            df=df,
            max_steps=5,
            answer_format=q0["format"],
        )
        results["agent_answers"]["mutual_info"] = answer
        # Parse: @top5_mi[name1, name2, name3, name4, name5]
        match = re.search(r"@top5_mi\[([^\]]+)\]", answer)
        if match:
            names = [n.strip() for n in match.group(1).split(",")]
            results["mutual_info_top5"] = names[:5]
        else:
            raise ValueError(f"Could not parse agent answer: {answer}")
    except Exception as e:
        print(f"Agent failed for mutual_info (ID 0): {e}")
        print("Using manual fallback...")
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_series = pd.Series(mi_scores, index=feature_names).sort_values(
            ascending=False
        )
        results["mutual_info_top5"] = mi_series.head(5).index.tolist()
        results["agent_answers"]["mutual_info"] = f"FALLBACK: {e}"

    # Question ID 1: F-regression
    q1 = questions[1]
    try:
        print(f"Running agent for Q24 ID 1: {q1['question'][:80]}...")
        answer, history = agent.run(
            question=q1["question"],
            constraints=q1["constraints"],
            df=df,
            max_steps=5,
            answer_format=q1["format"],
        )
        results["agent_answers"]["f_regression"] = answer
        match = re.search(r"@top5_f\[([^\]]+)\]", answer)
        if match:
            names = [n.strip() for n in match.group(1).split(",")]
            results["f_regression_top5"] = names[:5]
        else:
            raise ValueError(f"Could not parse agent answer: {answer}")
    except Exception as e:
        print(f"Agent failed for f_regression (ID 1): {e}")
        print("Using manual fallback...")
        f_scores, _ = f_regression(X, y)
        f_series = pd.Series(f_scores, index=feature_names).sort_values(ascending=False)
        results["f_regression_top5"] = f_series.head(5).index.tolist()
        results["agent_answers"]["f_regression"] = f"FALLBACK: {e}"

    save_json(results, "q24_top_features.json")

    # Create diamonds_selected.csv (ALL features — just copy standardized)
    shutil.copy2(STANDARDIZED_PATH, SELECTED_PATH)
    print(f"Saved: {SELECTED_PATH}")

    return agent


def run_q25_regression(agent):
    """Q25: Regression models using ReAct agent (OLS, Lasso, Ridge with 10-fold CV)."""
    print("=== Q25: Regression Models ===")

    questions = load_jsonl(QUESTIONS_PATH)
    df = pd.read_csv(SELECTED_PATH)

    X = df.drop(columns=["price"])
    y = df["price"]

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    results = {"agent_answers": {}}

    # Question ID 2: OLS
    q2 = questions[2]
    try:
        print(f"Running agent for Q25 ID 2: {q2['question'][:80]}...")
        answer, history = agent.run(
            question=q2["question"],
            constraints=q2["constraints"],
            df=df,
            max_steps=5,
            answer_format=q2["format"],
        )
        results["agent_answers"]["ols"] = answer
        match = re.search(r"@ols_val_rmse\[([^\]]+)\]", answer)
        if match:
            agent_rmse = float(match.group(1).strip())
            # Sanity check: RMSE should be reasonable given price scale (mean~3932, std~3989)
            # Agent often hallucinates unreasonably low values
            if agent_rmse < 100:
                raise ValueError(
                    f"Agent RMSE {agent_rmse} unreasonably low for price data (expected >100)"
                )
            results["ols"] = {"val_rmse": agent_rmse}
        else:
            raise ValueError(f"Could not parse agent answer: {answer}")
    except Exception as e:
        print(f"Agent failed for OLS (ID 2): {e}")
        print("Using manual fallback...")
        model = LinearRegression()
        neg_mse = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
        rmse = np.sqrt(-neg_mse).mean()
        results["ols"] = {"val_rmse": round(rmse, 2)}
        results["agent_answers"]["ols"] = f"FALLBACK: {e}"

    # Question ID 3: Lasso
    q3 = questions[3]
    try:
        print(f"Running agent for Q25 ID 3: {q3['question'][:80]}...")
        answer, history = agent.run(
            question=q3["question"],
            constraints=q3["constraints"],
            df=df,
            max_steps=5,
            answer_format=q3["format"],
        )
        results["agent_answers"]["lasso"] = answer
        alpha_match = re.search(r"@lasso_alpha\[([^\]]+)\]", answer)
        rmse_match = re.search(r"@lasso_val_rmse\[([^\]]+)\]", answer)
        if alpha_match and rmse_match:
            results["lasso"] = {
                "alpha": float(alpha_match.group(1).strip()),
                "val_rmse": float(rmse_match.group(1).strip()),
            }
        else:
            raise ValueError(f"Could not parse agent answer: {answer}")
    except Exception as e:
        print(f"Agent failed for Lasso (ID 3): {e}")
        print("Using manual fallback...")
        lasso_cv = LassoCV(cv=10, random_state=42, max_iter=10000)
        lasso_cv.fit(X, y)
        best_alpha = lasso_cv.alpha_
        model = Lasso(alpha=best_alpha, random_state=42)
        neg_mse = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
        rmse = np.sqrt(-neg_mse).mean()
        results["lasso"] = {
            "alpha": round(float(best_alpha), 4),
            "val_rmse": round(rmse, 2),
        }
        results["agent_answers"]["lasso"] = f"FALLBACK: {e}"

    # Question ID 4: Ridge
    q4 = questions[4]
    try:
        print(f"Running agent for Q25 ID 4: {q4['question'][:80]}...")
        answer, history = agent.run(
            question=q4["question"],
            constraints=q4["constraints"],
            df=df,
            max_steps=5,
            answer_format=q4["format"],
        )
        results["agent_answers"]["ridge"] = answer
        alpha_match = re.search(r"@ridge_alpha\[([^\]]+)\]", answer)
        rmse_match = re.search(r"@ridge_val_rmse\[([^\]]+)\]", answer)
        if alpha_match and rmse_match:
            results["ridge"] = {
                "alpha": float(alpha_match.group(1).strip()),
                "val_rmse": float(rmse_match.group(1).strip()),
            }
        else:
            raise ValueError(f"Could not parse agent answer: {answer}")
    except Exception as e:
        print(f"Agent failed for Ridge (ID 4): {e}")
        print("Using manual fallback...")
        ridge_cv = RidgeCV(alphas=np.logspace(-2, 4, 50), cv=10)
        ridge_cv.fit(X, y)
        best_alpha = ridge_cv.alpha_
        model = Ridge(alpha=best_alpha)
        neg_mse = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
        rmse = np.sqrt(-neg_mse).mean()
        results["ridge"] = {
            "alpha": round(float(best_alpha), 4),
            "val_rmse": round(rmse, 2),
        }
        results["agent_answers"]["ridge"] = f"FALLBACK: {e}"

    # Determine best model
    model_rmses = {
        name: results[name]["val_rmse"] for name in ["ols", "lasso", "ridge"]
    }
    results["best_model"] = min(model_rmses, key=model_rmses.get)

    save_json(results, "q25_regression_results.json")


if __name__ == "__main__":
    main()
