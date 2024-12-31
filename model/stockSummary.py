import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from scipy.stats import gaussian_kde
import numpy as np
# import seaborn as sns
# from datetime import datetime
# import matplotlib.pyplot as plt


def plot_distributions(symbol, start_date, end_date):
    # Download stock data
    stocks = yf.download(symbol, start=start_date, end=end_date)

    # Calculate log returns
    stocks["Log Returns"] = np.log(stocks["Adj Close"] / stocks["Adj Close"].shift(1))
    X = stocks["Log Returns"].dropna()

    # Calculate parameters
    mu_1 = 0.0005
    sigma1 = 0.022
    T = 2000

    # Generate mixture model data
    r = mu_1 + sigma1 * np.random.normal(0, 1, T)

    # Create plot using plotly
    fig = go.Figure()

    # Add empirical KDE
    kde_values = gaussian_kde(X)(np.linspace(X.min(), X.max(), 100))
    x_range = np.linspace(X.min(), X.max(), 100)

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=kde_values,
            name="Empirical Distribution",
            line=dict(color="blue", width=4),
        )
    )

    # Add mixture model as histogram
    fig.add_trace(
        go.Histogram(
            x=r,
            name="Mixture Model",
            nbinsx=100,
            histnorm="probability density",
            opacity=1,
            marker_color="green",
        )
    )

    fig.update_layout(
        title=f"{symbol} Returns Distribution (μ={mu_1:.6f}, σ={sigma1:.6f})",
        xaxis_title="Log Returns",
        yaxis_title="Density",
        xaxis_range=[-0.2, 0.2],
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        bargap=0.1,
    )
    return fig


def get_revenue_data(symbol):
    revenue_data = {
        "META": {
            "Family of Apps": 71.5,
            "Reality Labs": 1.2,
            "Other Revenue": 1.3,
            "WhatsApp Business": 2.0,
        },
        "GOOGL": {
            "Google Ads": 65.8,
            "Google Cloud": 12.5,
            "YouTube Ads": 13.2,
            "Google Network": 6.8,
            "Other Bets": 1.7,
        },
        "SNAP": {"Advertising": 92.5, "Snapchat+": 5.8, "Other Revenue": 1.7},
        "MSFT": {
            "Cloud (Azure)": 42.0,
            "Office & Business": 26.5,
            "Windows": 12.8,
            "Gaming & Xbox": 11.2,
            "LinkedIn": 5.5,
            "Other": 2.0,
        },
    }
    return revenue_data.get(symbol, {})


def plot_stock_statistics(symbol, period="1y"):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)

    stats_df = pd.DataFrame(
        {
            "Mean": hist["Close"].mean(),
            "Median": hist["Close"].median(),
            "Kurtosis": hist["Close"].kurtosis(),
            "Variance": hist["Close"].var(),
        },
        index=[0],
    )

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            name="Price",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=[hist["Close"].mean()] * len(hist),
            name="Mean",
            line=dict(color="red", dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=[hist["Close"].median()] * len(hist),
            name="Median",
            line=dict(color="green", dash="dash"),
        )
    )

    fig.update_layout(
        title=f"{symbol} Stock Price and Statistics",
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        height=400,  # Reduced height
        width=600,  # Set specific width
        margin=dict(l=0, r=0, t=30, b=0),  # Reduce margins
    )

    return fig, stats_df, stock.info


def format_number(num):
    if isinstance(num, str) or num is None:
        return "N/A"
    if num >= 1e12:
        return f"${num/1e12:.2f}T"
    if num >= 1e9:
        return f"${num/1e9:.2f}B"
    if num >= 1e6:
        return f"${num/1e6:.2f}M"
    if num >= 1e3:
        return f"${num/1e3:.2f}K"
    return f"${num:.2f}"


def format_percent(num):
    if isinstance(num, str) or num is None:
        return "N/A"
    return f"{num*100:.2f}%"


def create_category_table(stock_info, category_metrics):
    df = pd.DataFrame(
        {
            "Metric": [metric[0] for metric in category_metrics],
            "Value": [
                metric[1](stock_info.get(metric[2], "N/A"))
                for metric in category_metrics
            ],
        }
    )
    return df


def statistics_summary():
    symbols = ["META", "GOOGL", "MSFT", "SNAP"]

    # Create main layout
    col_left, col_right = st.columns([0.55, 0.45])  # Adjust ratio as needed
    with col_left:
        config1, config2 = st.columns(2)
        with config1:
            selected_stock = st.selectbox("Select Stock:", symbols)
        with config2:
            period = st.selectbox("Select Time Period:", ["3mo", "6mo", "2y", "5y"])

        fig, stats_df, stock_info = plot_stock_statistics(selected_stock, period)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        revenue_data = get_revenue_data(selected_stock)
        if revenue_data:
            fig_pie = px.pie(
                values=list(revenue_data.values()),
                names=list(revenue_data.keys()),
                title=f"{selected_stock} Revenue Streams",
                hole=0.3,
            )
            fig_pie.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

    valuation_metrics = [
        ("Market Cap", format_number, "marketCap"),
        ("Enterprise Value", format_number, "enterpriseValue"),
        ("P/E Ratio", lambda x: str(x), "trailingPE"),
        ("Forward P/E", lambda x: str(x), "forwardPE"),
        ("Price/Book", lambda x: str(x), "priceToBook"),
    ]

    trading_metrics = [
        ("Current Price", format_number, "currentPrice"),
        ("52 Week High", format_number, "fiftyTwoWeekHigh"),
        ("52 Week Low", format_number, "fiftyTwoWeekLow"),
        ("52 Week Change", format_percent, "52WeekChange"),
        ("Beta", lambda x: str(x), "beta"),
        ("Avg Volume", lambda x: f"{x/1e6:.2f}M", "averageVolume"),
    ]

    financial_metrics = [
        ("Revenue Growth", format_percent, "revenueGrowth"),
        ("Profit Margin", format_percent, "profitMargins"),
        ("ROE", format_percent, "returnOnEquity"),
        ("ROA", format_percent, "returnOnAssets"),
        ("Debt/Equity", lambda x: str(x), "debtToEquity"),
        ("Current Ratio", lambda x: str(x), "currentRatio"),
    ]

    analyst_metrics = [
        ("Target Price Mean", format_number, "targetMeanPrice"),
        ("Target Price Low", format_number, "targetLowPrice"),
        ("Target Price High", format_number, "targetHighPrice"),
        ("Analyst Count", lambda x: str(x), "numberOfAnalystOpinions"),
        (
            "Recommendation",
            lambda x: x.replace("_", " ").title(),
            "recommendationKey",
        ),
    ]
    st.markdown(
        """
            <style>
                .stDataFrame {
                    margin-bottom: 10px;
                }
                .stDataFrame td, .stDataFrame th {
                    padding: 3px;
                }
            </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("Valuation Metrics")
        st.dataframe(
            create_category_table(stock_info, valuation_metrics),
            height=200,
        )

    with col2:
        st.write("Trading Info")
        st.dataframe(create_category_table(stock_info, trading_metrics), height=200)

    with col3:
        st.write("Financial Metrics")
        st.dataframe(create_category_table(stock_info, financial_metrics), height=200)

    with col4:
        st.write("Analyst Coverage")
        st.dataframe(create_category_table(stock_info, analyst_metrics), height=200)

    st.subheader("KDE Plot for META Training Data")
    start_date2 = "2020-01-01"
    end_date2 = "2024-08-31"
    kde_fig = plot_distributions("META", start_date2, end_date2)
    st.plotly_chart(kde_fig, use_container_width=True)
