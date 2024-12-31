import streamlit as st


def tradingRules():
    st.markdown(
        """<div style=" font-size: 18px; line-height: 1.6;">
        The trading logic operates on a day-by-day prediction basis where the XGBoost model forecasts the next day's log return using a comprehensive set of market and economic features.
        <ul>
            <li><b>Buy Signal:</b> If predicted return &gt; 0.005 and confidence &gt; 0.01, a long position of 100 units is taken.</li>
            <li><b>Sell Signal:</b> If predicted return &lt; -0.005 with sufficient confidence, a short position is triggered.</li>
        </ul>
        Each trade is protected by a <b style="color: red;">1% stop-loss</b> and a <b style="color: green;">5% take-profit</b> level to secure gains.
        </div>""",
        unsafe_allow_html=True,
    )

    strat_col, risk_col = st.columns(2)

    with strat_col:
        st.subheader("Strategy Variables")
        st.markdown(
            """
            <ul>
                <li><b>Position size:</b> Fixed at 100 shares</li>
                <li><b>Capital:</b> $50,000</li>
                <li><b>Confidence threshold:</b> 0.01 (minimum predicted return to trigger a trade)</li>
                <li><b>Trading threshold:</b> 0.005 (determines buy/sell signals)</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

    with risk_col:
        st.subheader("Risk Management")
        st.markdown(
            """
            <ul>
                <li><b style="color: red;">Stop-loss:</b> 1% (limits maximum loss per trade)</li>
                <li><b style="color: green;">Take-profit:</b> 5% (locks in profits at predetermined level)</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

    st.code(
        """
    # Define stop-loss, take-profit, and confidence threshold levels
    stop_loss = 0.01  # 1% stop-loss
    take_profit = 0.05  # 5% take-profit
    confidence_threshold = 0.01  # Confidence threshold for making trades
    threshold = 0.005  # Prediction threshold for buy/sell signal
    
    # Trading simulation setup
    start_date_index = 252
    initial_balance = 50000
    balance = {start_date_index: initial_balance}
    position = 100
    signal = {}
    gain_loss = {}

    # Trading Logic loop
    for today in range(start_date_index, len(data_frame) - 1):
        try:
            # Prepare test data for prediction
            X_test_today = data_frame.iloc[today : today + 1][features]

            # Predict the log return for tomorrow
            y_pred_tomorrow = model.predict(X_test_today)[0]  # Added [0] to get scalar value

            # Confidence check before trading
            if abs(y_pred_tomorrow) > confidence_threshold:
                if y_pred_tomorrow > threshold:
                    signal[today + 1] = 1  # Buy signal
                    trade_gain_loss = (data_frame["Close"].iloc[today + 1] - data_frame["Open"].iloc[today + 1]) * position

                    # Implement stop-loss and take-profit check
                    if (data_frame["Close"].iloc[today + 1] - data_frame["Open"].iloc[today + 1]) / data_frame["Open"].iloc[today + 1] < -stop_loss:
                        trade_gain_loss = -stop_loss * position  # Stop-loss triggered
                        
                    elif (data_frame["Close"].iloc[today + 1] - data_frame["Open"].iloc[today + 1]) / data_frame["Open"].iloc[today + 1] > take_profit:
                        trade_gain_loss = take_profit * position  # Take-profit triggered

                elif y_pred_tomorrow < - threshold:
                    signal[today + 1] = -1  # Sell signal
                    trade_gain_loss = (data_frame["Open"].iloc[today + 1] - data_frame["Close"].iloc[today + 1]) * position

            # Update balance (only if no error occurs)
            prev_balance = balance[today]  # Use direct indexing
            balance[today + 1] = prev_balance + trade_gain_loss
            gain_loss[today + 1] = trade_gain_loss
        """,
        language="python",
    )
