import streamlit as st

images = {
    "Ridge Regression": "data/ridge-performance.png",
    "Random Forest": "data/randomforest-performance.png",
    "XGBoost": "data/xgboost-performance.png",
}

# Define performance metrics
performance_data = {
    "Ridge Regression": {"Profits": "$27,032.95", "Returns": "54.07%"},
    "Random Forest": {"Profits": "$33,592.97", "Returns": "67.19%"},
    "XGBoost": {"Profits": "40,401.00", "Returns": "80.80%"},
    "Buy & Hold": {"Profits": "$106,580.85", "Returns": "213.16%"},
}

positions = {
    "Ridge Regression": {"Long": "120", "Short": "300"},
    "Random Forest": {"Long": "120", "Short": "300"},
    "XGBoost": {"Long": "120", "Short": "300"},
    "Buy & Hold": {"Long": "120", "Short": "300"},
}


def tradingSignals():
    st.subheader("📊 Model Features")

    st.code("""
            features = ['RSI','RF','CMA','HML','Volatility',
                        'VIXCLS','RMW','SMB','DHHNGSP','EMVELECTGOVRN_interp']
                """)

    # Model selection with better styling
    st.subheader("Trading Model Comparison")
    method = st.selectbox("", ["Ridge Regression", "Random Forest", "XGBoost"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📈 Buy & Hold Strategy")
        st.metric("Profits", performance_data["Buy & Hold"]["Profits"], "")
        st.metric("Returns", performance_data["Buy & Hold"]["Returns"], "")

    with col2:
        st.subheader(f"🤖 {method}")
        st.metric("Profits", performance_data[method]["Profits"], "")
        st.metric("Returns", performance_data[method]["Returns"], "")
    with col3:
        st.subheader("💸 Trade Positions taken")
        st.metric("Long", positions[method]["Long"], "")
        st.metric("Short", positions[method]["Short"], "")
    st.markdown("</div>", unsafe_allow_html=True)

    # Display performance image
    st.image(images[method], caption=f"{method} Performance")

    # Key insights section
    st.subheader("🔑 Key Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            #### Model Performance Ranking:
              XGBoost **>** Random Forest **>** Ridge
            
            #### Model Sensitivity:
              Different models show varying sensitivity 
              to market conditions
        """)

    with col2:
        st.markdown("""
            #### Trading Strategy Impact:
              Active trading doesn't guarantee higher profits
            
            #### Signal Quality:
              Quality of trading signals matters more than quantity
        """)
