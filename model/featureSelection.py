# analysis.py
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import shap
import plotly.express as px
import matplotlib.pyplot as plt

# import plotly.figure_factory as ff
# import plotly.graph_objects as go
# import seaborn as sns


class ModelAnalyzer:
    def __init__(self, data, target_col):
        self.data = data
        self.target_col = target_col
        self.X = None
        self.y = None
        self.model = None
        self.shap_values = None
        self.vif_results = None
        self.corr_matrix = None
        self.train_rmse = None
        self.test_rmse = None

    def prepare_data(self):
        # Prepare features
        self.X = self.data.drop(columns=[self.target_col, "Date"])
        self.X = self.X.select_dtypes(include=["int64", "float64"])
        self.X = self.X.replace([np.inf, -np.inf], np.nan)
        self.X = self.X.fillna(self.X.mean())
        self.y = self.data[self.target_col]

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.model = XGBRegressor(
            learning_rate=0.1, max_depth=3, n_estimators=100, random_state=42
        )

        self.model.fit(X_train, y_train)
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)

        self.train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        self.test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    def calculate_vif(self):
        X_vif = add_constant(self.X)
        self.vif_results = pd.DataFrame(
            {
                "Feature": X_vif.columns,
                "VIF": [
                    variance_inflation_factor(X_vif.values, i)
                    for i in range(X_vif.shape[1])
                ],
            }
        )
        # Remove the constant term (first row)
        self.vif_results = self.vif_results.iloc[0:]

    def calculate_correlations(self):
        self.corr_matrix = self.X.corr()

    def calculate_shap(self):
        # Make sure self.model and self.X are defined
        explainer = shap.Explainer(self.model)
        self.shap_values = explainer(self.X)

        # Store the feature names
        self.feature_names = self.X.columns.tolist()

    def analyze(self):
        self.prepare_data()
        self.train_model()
        self.calculate_vif()
        self.calculate_correlations()
        self.calculate_shap()
        return {
            "train_rmse": self.train_rmse,
            "test_rmse": self.test_rmse,
            "vif_results": self.vif_results,
            "corr_matrix": self.corr_matrix,
            "shap_values": self.shap_values,
            "features": self.X,
            "model": self.model,
        }


class ModelVisualizer:
    def __init__(self, analysis_results):
        self.results = analysis_results

    def plot_rmse_comparison(self):
        st.subheader("Model Performance - RMSE")
        rmse_data = pd.DataFrame(
            {
                "Metric": ["Train RMSE", "Test RMSE"],
                "Value": [self.results["train_rmse"], self.results["test_rmse"]],
            }
        )
        fig = px.bar(rmse_data, x="Metric", y="Value")
        st.plotly_chart(fig)

    def plot_vif_results(self):
        st.subheader("Variance Inflation Factors")
        fig = px.bar(self.results["vif_results"], x="Feature", y="VIF")
        # Customize the layout
        fig.update_layout(
            xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
            height=600,  # Adjust height of the plot
            margin=dict(b=100),  # Add bottom margin for rotated labels
        )
        st.plotly_chart(fig)

    def plot_correlation_matrix(self):
        st.subheader("Feature Correlation Matrix")

        # Create a larger figure with annotations for better readability
        fig = px.imshow(
            self.results["corr_matrix"],
            color_continuous_scale="rdbu_r",
            aspect="auto",  # Ensures the plot uses available space
            labels=dict(color="Correlation"),
            zmin=-1,
            zmax=1,  # Set color scale limits for correlation
        )

        # Update layout for better readability
        fig.update_layout(
            width=800,  # Set a larger width
            height=800,  # Set a larger height
            margin=dict(l=100, r=100, t=100, b=100),  # Add margins
            xaxis_title="Features",
            yaxis_title="Features",
        )

        st.plotly_chart(fig)

    def plot_shap_summary(self):
        st.subheader("SHAP Feature Importance")

        # Adjust the figure size to make it smaller
        fig, ax = plt.subplots(figsize=(8, 6))  # Smaller size

        # Generate the SHAP summary plot
        shap.summary_plot(
            self.results["shap_values"],
            self.results["features"],
            show=False,
            plot_size=(8, 6),  # Adjust plot size within SHAP
        )

        # Display the plot in Streamlit
        st.pyplot(fig)

    def display_all_plots(self):
        st.subheader("Remove features with high VIF")
        self.plot_vif_results()
        st.code("""
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

                """)
        self.plot_correlation_matrix()
        st.code("""
    # Remove highly correlated features
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
                """)
        self.plot_shap_summary()
        st.code("""
                
                """)
