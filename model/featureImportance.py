import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import streamlit as st
import xgboost as xgb


def ridge_regression():
    # Step 1: Download Meta stock data for the specified time frame
    meta_data = yf.download("META", start="2020-01-01", end="2024-08-30", interval="1d")

    # Reset index to make 'Date' a column
    meta_data.reset_index(inplace=True)

    # Calculate log returns for Meta stock
    meta_data["Log Return"] = np.log(meta_data["Close"] / meta_data["Close"].shift(1))
    meta_data = (
        meta_data.dropna()
    )  # Drop rows with NaN values caused by the shift operation

    # Keep only relevant columns: Date and Log Return
    meta_data = meta_data[["Date", "Log Return"]]

    # Step 2: Load the provided dataset (3.csv)
    file_path = "/Users/srushtipatil/Desktop/Northeastern University/Data Sciece/final_dashboard/data/3.csv"  # Update if necessary
    data = pd.read_csv(file_path)

    # Convert the 'Date' column to datetime for both datasets
    data["Date"] = pd.to_datetime(data["Date"])
    meta_data["Date"] = pd.to_datetime(meta_data["Date"])

    # Drop 'log_return' column from 3.csv if it exists
    if "log_return" in data.columns:
        data = data.drop(columns=["log_return"])

    # Flatten the multi-level columns in meta_data
    meta_data.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}" for col in meta_data.columns
    ]

    # Step 3: Align datasets by date using an inner join
    merged_data = pd.merge(data, meta_data, on="Date", how="inner")

    # Handle missing values in features by imputing with the mean
    merged_data.fillna(merged_data.mean(), inplace=True)

    # Step 4: Define features (X) and target (y)
    X = merged_data.drop(columns=["Date", "Log Return"])  # Drop irrelevant columns
    y = merged_data["Log Return"]

    # Step 5: Eliminate problematic features
    # 5.1 Remove features with high VIF
    X_vif = add_constant(X)  # Add constant for intercept
    vif = pd.DataFrame()
    vif["Feature"] = X_vif.columns
    vif["VIF"] = [
        variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])
    ]
    high_vif_features = vif[vif["VIF"] > 10]["Feature"].tolist()
    X = X.drop(columns=high_vif_features, errors="ignore")

    # 5.2 Remove low variance features
    low_variance_features = X.columns[X.var() < 0.01].tolist()
    X = X.drop(columns=low_variance_features, errors="ignore")

    # 5.3 Remove highly correlated features
    correlation_threshold = 0.8
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_features = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > correlation_threshold)
    ]
    X = X.drop(columns=high_corr_features, errors="ignore")

    # Step 6: Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 7: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Step 8: Tune hyperparameters using GridSearchCV
    ridge_params = {"alpha": [0.01, 0.1, 1, 10, 100]}
    ridge_model = Ridge()
    grid_search = GridSearchCV(
        ridge_model, ridge_params, scoring="neg_mean_squared_error", cv=5
    )
    grid_search.fit(X_train, y_train)

    best_alpha = grid_search.best_params_["alpha"]
    best_score = np.sqrt(-grid_search.best_score_)

    print(f"Best Alpha: {best_alpha}")
    print(f"Best Cross-Validation RMSE: {best_score}")

    # Step 9: Train Ridge Regression with the best hyperparameter
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_train, y_train)

    # Step 10: Evaluate the model
    train_preds = final_model.predict(X_train)
    test_preds = final_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    print(f"Train RMSE: {train_rmse}")
    print(f"Test RMSE: {test_rmse}")

    # Step 11: Visualize coefficient magnitudes for feature importance
    coefficients = pd.DataFrame(
        {"Feature": X.columns, "Coefficient": final_model.coef_}
    )
    coefficients = coefficients.sort_values(by="Coefficient", ascending=False)

    fig = plt.figure(figsize=(12, 8))
    plt.barh(coefficients["Feature"], coefficients["Coefficient"], color="purple")
    plt.title("Ridge Regression Coefficients (Feature Importance)", fontsize=16)
    plt.xlabel("Coefficient Value", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)


def xgboostfunc():
    # Step 1: Download Meta stock data for the specified time frame
    meta_data = yf.download("META", start="2020-01-01", end="2024-08-30", interval="1d")

    # Reset index to make 'Date' a column
    meta_data.reset_index(inplace=True)

    # Calculate log returns for Meta stock
    meta_data["Log Return"] = np.log(meta_data["Close"] / meta_data["Close"].shift(1))
    meta_data = (
        meta_data.dropna()
    )  # Drop rows with NaN values caused by the shift operation

    # Keep only relevant columns: Date and Log Return
    meta_data = meta_data[["Date", "Log Return"]]

    # Step 2: Load the provided dataset (3.csv)
    file_path = "/Users/srushtipatil/Desktop/Northeastern University/Data Sciece/final_dashboard/data/3.csv"  # Update if necessary
    data = pd.read_csv(file_path)

    # Convert the 'Date' column to datetime for both datasets
    data["Date"] = pd.to_datetime(data["Date"])
    meta_data["Date"] = pd.to_datetime(meta_data["Date"])

    # Drop 'log_return' column from 3.csv if it exists
    if "log_return" in data.columns:
        data = data.drop(columns=["log_return"])

    # Flatten the multi-level columns in meta_data
    meta_data.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}" for col in meta_data.columns
    ]

    # Step 3: Align datasets by date using an inner join
    merged_data = pd.merge(data, meta_data, on="Date", how="inner")

    # Handle missing values in features by imputing with the mean
    merged_data.fillna(merged_data.mean(), inplace=True)

    # Step 4: Define features (X) and target (y)
    X = merged_data.drop(columns=["Date", "Log Return"])  # Drop irrelevant columns
    y = merged_data["Log Return"]

    # Step 5: Eliminate problematic features
    # 5.1 Remove features with high VIF
    X_vif = add_constant(X)  # Add constant for intercept
    vif = pd.DataFrame()
    vif["Feature"] = X_vif.columns
    vif["VIF"] = [
        variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])
    ]
    high_vif_features = vif[vif["VIF"] > 10]["Feature"].tolist()
    X = X.drop(columns=high_vif_features, errors="ignore")

    # 5.2 Remove low variance features
    low_variance_features = X.columns[X.var() < 0.01].tolist()
    X = X.drop(columns=low_variance_features, errors="ignore")

    # 5.3 Remove highly correlated features
    correlation_threshold = 0.8
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_features = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > correlation_threshold)
    ]
    X = X.drop(columns=high_corr_features, errors="ignore")

    # Step 6: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 7: Initialize the XGBoost model with tuned hyperparameters
    xgb_model = xgb.XGBRegressor(
        learning_rate=0.05, max_depth=3, n_estimators=200, random_state=42
    )

    # Train the model
    xgb_model.fit(X_train, y_train)

    # Step 8: Evaluate the model and calculate RMSE
    train_preds = xgb_model.predict(X_train)
    test_preds = xgb_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    print(f"Train RMSE with tuned hyperparameters: {train_rmse}")
    print(f"Test RMSE with tuned hyperparameters: {test_rmse}")

    # Extract feature importance
    feature_importances_gain = xgb_model.get_booster().get_score(importance_type="gain")
    feature_importances_fscore = xgb_model.get_booster().get_score(
        importance_type="weight"
    )

    # Convert importance dictionaries to pandas DataFrame for visualization
    gain_importance_df = pd.DataFrame(
        feature_importances_gain.items(), columns=["Feature", "Importance (Gain)"]
    ).sort_values(by="Importance (Gain)", ascending=False)
    fscore_importance_df = pd.DataFrame(
        feature_importances_fscore.items(), columns=["Feature", "Importance (F-Score)"]
    ).sort_values(by="Importance (F-Score)", ascending=False)

    # Step 9: Visualize feature importance (Gain)
    fig1 = plt.figure(figsize=(12, 8))
    plt.barh(
        gain_importance_df["Feature"],
        gain_importance_df["Importance (Gain)"],
        color="skyblue",
    )
    plt.title("XGBoost Feature Importance (Gain)", fontsize=16)
    plt.xlabel("Importance (Gain)", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig1)

    # Step 10: Visualize feature importance (F-Score)
    fig2 = plt.figure(figsize=(12, 8))
    plt.barh(
        fscore_importance_df["Feature"],
        fscore_importance_df["Importance (F-Score)"],
        color="lightcoral",
    )
    plt.title("XGBoost Feature Importance (F-Score)", fontsize=16)
    plt.xlabel("Importance (F-Score)", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig2)
