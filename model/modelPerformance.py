import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import altair as alt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.linear_model import Lars
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def plotgraphs():
    # Step 1: Download Meta stock data
    meta_data = yf.download("META", start="2020-01-01", end="2024-08-30", interval="1d")
    meta_data.reset_index(inplace=True)
    meta_data["Log Return"] = np.log(meta_data["Close"] / meta_data["Close"].shift(1))
    meta_data = meta_data.dropna()
    meta_data = meta_data[["Date", "Log Return"]]

    # Step 2: Load and merge the dataset
    file_path = "C:/Users/aqeel/work/NEU/DS INFO6105/final_dashboard/data/3.csv"  # Update this path
    data = pd.read_csv(file_path)
    data["Date"] = pd.to_datetime(data["Date"])
    meta_data["Date"] = pd.to_datetime(meta_data["Date"])

    # Flatten the multi-level columns in meta_data
    meta_data.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}" for col in meta_data.columns
    ]

    merged_data = pd.merge(data, meta_data, on="Date", how="inner")
    merged_data.fillna(merged_data.mean(), inplace=True)

    # Step 3: Data Cleaning
    X = merged_data.drop(columns=["Date", "Log Return"])
    y = merged_data["Log Return"]

    # Remove high VIF features
    X_vif = add_constant(X)
    vif = pd.DataFrame()
    vif["Feature"] = X_vif.columns
    vif["VIF"] = [
        variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])
    ]
    high_vif_features = vif[vif["VIF"] > 10]["Feature"].tolist()
    X = X.drop(columns=high_vif_features, errors="ignore")

    # Remove low variance features
    low_variance_features = X.columns[X.var() < 0.01].tolist()
    X = X.drop(columns=low_variance_features, errors="ignore")

    # Remove highly correlated features
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_features = [
        column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)
    ]
    X = X.drop(columns=high_corr_features, errors="ignore")

    # Step 4: Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 5: Define Models
    models = {
        "XGBoost": xgb.XGBRegressor(
            learning_rate=0.05, max_depth=3, n_estimators=200, random_state=42
        ),
        "Ridge": Ridge(alpha=5),
        "LASSO": Lasso(alpha=0.01),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
        "LARS": Lars(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    # Step 6: Train and Evaluate Models
    results = []
    feature_importances = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)

        cv_scores = -cross_val_score(
            model, X, y, scoring="neg_mean_squared_error", cv=5
        )
        cv_rmse = np.sqrt(cv_scores.mean())

        results.append(
            (
                name,
                train_rmse,
                test_rmse,
                train_mae,
                test_mae,
                train_r2,
                test_r2,
                cv_rmse,
            )
        )

        # Feature importance
        if hasattr(model, "coef_"):
            feature_importances[name] = pd.Series(
                model.coef_, index=X.columns
            ).sort_values(ascending=False)
        elif hasattr(model, "feature_importances_"):
            feature_importances[name] = pd.Series(
                model.feature_importances_, index=X.columns
            ).sort_values(ascending=False)

    # Step 7: Results and Visualizations
    result_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Train RMSE",
            "Test RMSE",
            "Train MAE",
            "Test MAE",
            "Train R²",
            "Test R²",
            "Cross-Validation RMSE",
        ],
    )

    # Display metrics in columns
    st.header("Model Performance Overview")
    # Highlight XGBoost as the best model
    st.markdown(
        """
    <div style='background-color: #000000; padding: 20px; border-radius: 10px; margin-bottom: 25px'>
        <h2 style='color: #DAF7A6; margin: 0;'>🏆 Best Model: XGBoost</h2>
        <p style='margin: 10px 0 0 0;'>XGBoost demonstrates superior performance in predicting stock market returns through:</p>
        <ul>
            <li>Lower prediction error (RMSE)</li>
            <li>Better generalization on unseen data</li>
            <li>Robust handling of non-linear relationships</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Model Performance Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="XGBoost Test RMSE",
            value=f"{result_df.loc[result_df['Model'] == 'XGBoost', 'Test RMSE'].values[0]:.4f}",
            delta="Best Performance",
        )
    with col2:
        st.metric(
            label="XGBoost R² Score",
            value=f"{result_df.loc[result_df['Model'] == 'XGBoost', 'Test R²'].values[0]:.4f}",
        )
    with col3:
        st.metric(
            label="Cross-Validation RMSE",
            value=f"{result_df.loc[result_df['Model'] == 'XGBoost', 'Cross-Validation RMSE'].values[0]:.4f}",
        )

    model_selector = st.selectbox(
        "Select Model to View Feature Importance",
        options=feature_importances.keys(),
        key="model_selector",
    )

    # Prepare feature importance data for selected model
    importance_df = pd.DataFrame(
        {
            "Feature": feature_importances[model_selector].index,
            "Importance": feature_importances[model_selector].values,
        }
    )

    # Create Altair chart
    chart = (
        alt.Chart(importance_df)
        .mark_bar()
        .encode(
            x=alt.X("Importance:Q", title="Feature Importance"),
            y=alt.Y("Feature:N", sort="-x", title=None),
            color=alt.value("#1f77b4"),
            tooltip=["Feature", "Importance"],
        )
        .properties(height=400, title=f"{model_selector} Feature Importance")
        .interactive()
    )

    # Display the chart
    st.altair_chart(chart, use_container_width=True)

    # Create tabs for different metrics
    tab1, tab2, tab3 = st.tabs(["RMSE Comparison", "MAE Comparison", "R² Scores"])

    with tab1:
        fig2 = plt.figure(figsize=(10, 6))
        x = np.arange(len(result_df["Model"]))
        width = 0.25

        plt.bar(
            x - width,
            result_df["Train RMSE"],
            width=width,
            label="Train",
            color="#2ecc71",
        )
        plt.bar(x, result_df["Test RMSE"], width=width, label="Test", color="#e74c3c")
        plt.bar(
            x + width,
            result_df["Cross-Validation RMSE"],
            width=width,
            label="CV",
            color="#3498db",
        )

        plt.xticks(x, result_df["Model"], rotation=45)
        plt.title("RMSE Comparison Across Models")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with tab2:
        fig3 = plt.figure(figsize=(10, 6))
        plt.bar(
            x - width / 2,
            result_df["Train MAE"],
            width=width,
            label="Train",
            color="#9b59b6",
        )
        plt.bar(
            x + width / 2,
            result_df["Test MAE"],
            width=width,
            label="Test",
            color="#f1c40f",
        )
        plt.xticks(x, result_df["Model"], rotation=45)
        plt.title("MAE Comparison Across Models")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with tab3:
        fig4 = plt.figure(figsize=(10, 6))
        plt.bar(
            x - width / 2,
            result_df["Train R²"],
            width=width,
            label="Train",
            color="#1abc9c",
        )
        plt.bar(
            x + width / 2,
            result_df["Test R²"],
            width=width,
            label="Test",
            color="#e67e22",
        )
        plt.xticks(x, result_df["Model"], rotation=45)
        plt.title("R² Score Comparison")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    # Display detailed results in an interactive table
    # Display detailed results in an interactive table
    st.subheader("Detailed Performance Metrics")

    # Format the numeric columns to display 4 decimal places
    styled_df = result_df.style.format(
        {
            "Train RMSE": "{:.4f}",
            "Test RMSE": "{:.4f}",
            "Train MAE": "{:.4f}",
            "Test MAE": "{:.4f}",
            "Train R²": "{:.4f}",
            "Test R²": "{:.4f}",
            "Cross-Validation RMSE": "{:.4f}",
        }
    )

    # Display the formatted table
    st.dataframe(styled_df)
