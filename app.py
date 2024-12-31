import streamlit as st
import pandas as pd
from model.stockSummary import statistics_summary
from model.featureDatabase import FeatureDatabase
from model.featureSelection import ModelVisualizer, ModelAnalyzer
from model.featureImportance import ridge_regression, xgboostfunc
from model.modelPerformance import plotgraphs

# from model.benchmark import benchmark_study
from model.tradingrule import tradingRules
from model.modelcomparision import tradingSignals


def load_data():
    file_path = (
        "data/INFO6105_FeatureMart_with_meta_indicators.csv"  # Update if necessary
    )
    data = pd.read_csv(file_path)

    # Convert the 'Date' column to datetime for both datasets
    data["Date"] = pd.to_datetime(data["Date"])
    # Handle missing values and reset the index
    data.fillna(method="ffill", inplace=True)

    return data


# Set page configuration
st.set_page_config(page_title="Final project Dashboard")

# Create sidebar
st.sidebar.title("Stock Analysis Dashboard")

# Create sidebar options
sidebar_options = [
    "1. Stock Summary",
    "2. Feature Database",
    "3. Feature Selection",
    "4. Model Performance",
    "5. Benchmark Study",
    "6. Trading Rules",
    "7. Model Comparison",
]

# Create radio buttons for sidebar navigation
selected_task = st.sidebar.radio("Select a task:", sidebar_options)

# Main content area
# st.title("Stock Analysis Dashboard")

# Display content based on selected task
if selected_task == "1. Stock Summary":
    st.header("Stock Summary")
    statistics_summary()

elif selected_task == "2. Feature Database":
    st.header("Feature Database with Meta Indicators")
    # FAMA FRENCH DATA--------------------------------------------
    st.subheader("Fama-French 5 factors (Daily Data)")
    st.write(
        "The Fama/French 5 factors (2x3) are constructed using the 6 value-weight portfolios formed on size and book-to-market, the 6 value-weight portfolios formed on size and operating profitability, and the 6 value-weight portfolios formed on size and investment."
    )
    code = """
        column_names=['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        
        data_ff5 = pd.read_csv
            ('F-F_Research_Data_5_Factors_2x3_daily.csv',usecols=range(7),
            names=column_names,
            header=None,
            skiprows=4)
        
        data_ff5['date'] = data_ff5['date'].astype(str).str[0:4]+'-'+data_ff5['date'].astype(str).str[4:6]+'-'+data_ff5['date'].astype(str).str[6:8]
        data_ff5['date'] = pd.to_datetime(data_ff5['date'], format='%Y-%m-%d')
        data_ff5['date'] = data_ff5['date'].dt.date
        
        df_ffs = data_ff5.set_index('date')

        print(df_ffs.head())
        print(df_ffs.shape)"""
    st.code(code, language="python")

    # ADS INDEX--------------------------------------------
    st.subheader("ADS index")
    st.write(
        "The Aruoba-Diebold-Scotti Business Conditions Index (ADS Index) is a coincident business cycle indicator used in macroeconomics in the United States. The index measures business activity, which may be correlated with periods of expansion and contraction in the economy."
    )
    code = """
        data_ads = pd.read_excel('ADS_Index_Most_Current_Vintage.xlsx')
        # manually replace : into -
        df_ads = data_ads.set_index('date')
        df_ads.index = pd.to_datetime(df_ads.index, format='%Y:%m:%d')
        """
    st.code(code, language="python")

    # FRED DATA--------------------------------------------
    st.subheader("FRED Data")
    st.write(
        "FRED includes hundreds of thousands of economic time series from national, international, public, and private sources. The data covers major areas of macroeconomic analysis, including:"
    )
    code = """
        fred = Fred(api_key='API~KEY~HERE')
        varList = ['T10Y3M', 'DGS10', 'OBMMIJUMBO30YF',  # term premium 10yr-3mon, 30 yr mortgage jumbo loan
                'DEXUSEU', 'DEXJPUS', 'DEXUSUK', # spot exchange rates to EUR, JPY, GBP 
                'CBBTCUSD', 'CBETHUSD',  # cryptocurrencies
                'T10YIE', 'DCOILBRENTEU', # breakeven inflation + brent oil price 
                'VIXCLS', # implied volatilities
                'DAAA', 'DBAA', # corporate bond yield
                'AMERIBOR', 'T5YIE', 'BAMLH0A0HYM2',
                'BAMLH0A0HYM2EY', 'DGS1', 'DCOILWTICO', 
                'DHHNGSP'] 

        SP500 = fred.get_series('SP500')
        SP500.name = 'SP500'
        df_fred = SP500

        # merge data series
        for i in range(0, len(varList)):
            data = fred.get_series(varList[i])
            data.name = varList[i]
            df_fred = pd.merge(df_fred, data, left_index=True, right_index=True)
        """
    st.code(code, language="python")

    # TECH INDICATORS--------------------------------------------
    st.subheader("Technical Indicators for META")
    st.write("RSI, Bollinger Bands, SMA20, SMA50, On Balance Volume")
    code = """
        # Calculate RSI
        stock_data['RSI'] = RSIIndicator(close=stock_data['Close'], window=14).rsi()

        # Calculate Bollinger Band-based Volatility
        bb_indicator = BollingerBands(close=stock_data['Close'], window=20, window_dev=2)
        stock_data['Volatility'] = bb_indicator.bollinger_hband() - bb_indicator.bollinger_lband()

        # Calculate Simple Moving Averages (SMA20 and SMA50)
        stock_data['SMA20'] = SMAIndicator(close=stock_data['Close'], window=20).sma_indicator()
        stock_data['SMA50'] = SMAIndicator(close=stock_data['Close'], window=50).sma_indicator()

        # Calculate On-Balance Volume (OBV)
        stock_data['OBV'] = OnBalanceVolumeIndicator(close=stock_data['Close'], volume=stock_data['Volume']).on_balance_volume()
        """
    st.code(code, language="python")

    db = FeatureDatabase("data/INFO6105_FeatureMart_with_meta_indicators.csv")
    db.load_data()
    db.display_data()
    data = db.get_data()

elif selected_task == "3. Feature Selection":
    st.title("Feature Selection")
    data = load_data()
    analyzer = ModelAnalyzer(data, "log_return")
    analysis_results = analyzer.analyze()
    visualizer = ModelVisualizer(analysis_results)
    visualizer.display_all_plots()

    # Model selection
    st.header("Feature Importance")
    model_name = st.selectbox("", ["Ridge Regression", "XGBoost"])
    if model_name == "Ridge Regression":
        st.code("""
                # Tune hyperparameters using GridSearchCV
                ridge_params = {"alpha": [0.01, 0.1, 1, 10, 100]}
                ridge_model = Ridge()
                grid_search = GridSearchCV(
                    ridge_model, ridge_params, scoring="neg_mean_squared_error", cv=5
                )
                grid_search.fit(X_train, y_train)
                
                # Train Ridge Regression with the best hyperparameter
                final_model = Ridge(alpha=best_alpha)
                final_model.fit(X_train, y_train)
                """)
        ridge_regression()
    elif model_name == "XGBoost":
        st.code("""
            # XGBoost model with tuned hyperparameters
            xgb_model = xgb.XGBRegressor(
                learning_rate=0.05, 
                max_depth=3, 
                n_estimators=200, 
                random_state=42)

            # Model Training
            xgb_model.fit(X_train, y_train)
           """)
        xgboostfunc()

elif selected_task == "4. Model Performance":
    st.title("Model Performance Comparison")
    st.subheader("Target Variable: log_return")
    st.code("""
    # define daterange
    start_date = datetime(2019,3,14)
    end_date = datetime(2024,8,30)

    # prepare features
    feature = pd.read_csv('FeatureMart_with_meta_indicators.csv', index_col = [1])
    feature['Date'] = pd.to_datetime(feature['Date'])
    feature = feature.set_index('Date')

    # fetch target variable (stock price or stock returns)
    stock_symbol = 'META'
    stock = yf.download(stock_symbol, start_date, end_date)
    print(stock_symbol + ' Stock Price History')
    print(stock.head())

    # Correct the column names after merging
    stock.columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']

    # Reset the index before merging
    stock.reset_index(inplace=True)
    feature.reset_index(inplace=True)

    # Perform the merge
    data_frame = pd.merge(feature, stock, how='inner', on='Date')

    # Handle missing values and reset the index
    data_frame.fillna(method='ffill', inplace=True)
    data_frame.reset_index(drop=True, inplace=True)
    
    # Calculate log return using the previous day's Close price
    data_frame['log_return'] = np.log(data_frame['Close'] / data_frame['Close'].shift(1))

    # Drop missing values created by the shift operation
    data_frame.dropna(inplace=True)
            
            """)
    st.subheader("Quant Model structures")
    st.code("""
    models = {
        "XGBoost": xgb.XGBRegressor(learning_rate=0.05, max_depth=3, n_estimators=200, random_state=42),
        "Ridge": Ridge(alpha=5),
        "LASSO": Lasso(alpha=0.01),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
        "LARS": Lars(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    }
        """)
    plotgraphs()
    # Add code for task 4 here

elif selected_task == "5. Benchmark Study":
    # Title
    st.title("GARCH-t Model Analysis Results")
    st.code("""
def garch(param, *args):

    # Initialize Params
    mu = param[0]
    omega = param[1]
    alpha = param[2]
    beta = param[3]
    nv = param[4]
    GARCH_Dens, sigma2, F, v = {}, {}, {}, {}
    
    # Initialize values
    sigma2[0] = np.var(Y)
    Likelihood = 0
    for t in range(1, T):
        sigma2[t] = omega + alpha * ((Y[t - 1] - mu) ** 2) + beta * (sigma2[t - 1])
        if sigma2[t] < 0:
            sigma2[t] = 1e-2
        F[t] = Y[t] - mu - np.sqrt(sigma2[t]) * np.random.standard_t(nv, 1)
        v[t] = sigma2[t]
            GARCH_Dens[t] = (
                    np.log(ss.gamma((nv + 1) / 2))
                    - np.log(np.sqrt(nv * np.pi))
                    - np.log(ss.gamma(nv / 2))
                    - ((nv + 1) / 2) * np.log(1 + ((F[t] ** 2) / v[t]) / nv)
                )
            Likelihood += GARCH_Dens[t]

        return -Likelihood
        """)

    # Display images in columns
    st.image("data/logvspred.png", caption="Returns vs Predictions")
    st.image("data/resiTime.png", caption="Standardized Residuals")
    st.image("data/volaOvertime.png", caption="Volatility Over Time")

    # Create two columns for parameters and statistics
    params_col, stats_col = st.columns(2)

    with params_col:
        st.subheader("ARCH-t Model Parameters")
        params_df = pd.DataFrame(
            {
                "Parameter": [
                    "Mu (Mean)",
                    "Omega (Variance Constant)",
                    "Alpha (ARCH Effect)",
                    "Beta (GARCH Effect)",
                    "Nu (Degree of Freedom)",
                ],
                "Value": [0.0043, 0.0000, 1.2000, 0.2500, 10.0000],
            }
        )
        st.dataframe(params_df.style.format({"Value": "{:.4f}"}))

    with stats_col:
        st.subheader("Log returns summary")
        stats_data = {
            "Metric": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            "Value": [
                269.000000,
                0.004332,
                0.023401,
                -0.046546,
                -0.008727,
                0.002103,
                0.014972,
                0.209307,
            ],
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df.style.format({"Value": "{:.6f}"}))

elif selected_task == "6. Trading Rules":
    st.title("Trading Strategy Overview")
    tradingRules()
    # Add code for task 6 here

elif selected_task == "7. Model Comparison":
    st.title("Trading Signals and Profit/Loss Comparison")
    tradingSignals()
