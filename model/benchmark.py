def benchmark_study(start_date="2020-01-01", end_date="2023-12-31"):
    """
    Performs benchmark study of META stock using GARCH and Kalman Filter models

    Args:
        start_date (str): Start date for analysis in 'YYYY-MM-DD' format
        end_date (str): End date for analysis in 'YYYY-MM-DD' format

    Returns:
        dict: Dictionary containing model parameters and performance metrics
    """
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt

    # Download META data
    meta = yf.download("META", start=start_date, end=end_date)
    returns = 100 * meta["Adj Close"].pct_change().dropna().values
    T = len(returns)

    # GARCH model implementation
    def garch(params):
        mu, omega, alpha, beta = params
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(returns)
        llh = 0

        for t in range(1, T):
            sigma2[t] = (
                omega + alpha * ((returns[t - 1] - mu) ** 2) + beta * sigma2[t - 1]
            )
            llh += 0.5 * (
                np.log(2 * np.pi)
                + np.log(sigma2[t])
                + ((returns[t] - mu) ** 2) / sigma2[t]
            )
        return llh

    # Kalman Filter implementation
    def kalman_filter(params):
        Z, T, H, Q = params
        S = len(returns)
        u_predict = np.zeros(S)
        P_predict = np.zeros(S)
        u_update = np.zeros(S)
        P_update = np.zeros(S)
        v = np.zeros(S)
        F = np.zeros(S)

        u_update[0] = returns[0]
        u_predict[0] = u_update[0]
        P_update[0] = np.var(returns) / 4
        P_predict[0] = T * P_update[0] * T + Q

        llh = 0
        for s in range(1, S):
            F[s] = Z * P_predict[s - 1] * Z + H
            v[s] = returns[s] - Z * u_predict[s - 1]
            u_update[s] = u_predict[s - 1] + P_predict[s - 1] * Z * (1 / F[s]) * v[s]
            u_predict[s] = T * u_update[s]
            P_update[s] = (
                P_predict[s - 1]
                - P_predict[s - 1] * Z * (1 / F[s]) * Z * P_predict[s - 1]
            )
            P_predict[s] = T * P_update[s] * T + Q
            llh += 0.5 * (
                np.log(2 * np.pi) + np.log(abs(F[s])) + v[s] * (1 / F[s]) * v[s]
            )
        return llh

    # Optimize GARCH
    garch_param = minimize(
        garch, x0=[0, 0.1, 0.1, 0.8], method="BFGS", options={"disp": True}
    )

    # Optimize Kalman
    kalman_param = minimize(
        kalman_filter, x0=[1, 1, 0.1, 0.1], method="BFGS", options={"disp": True}
    )

    # Calculate predictions and RMSE
    def calculate_rmse(actual, predicted):
        return np.sqrt(np.mean((actual - predicted) ** 2))

    # Store results
    results = {
        "garch_parameters": garch_param.x,
        "kalman_parameters": kalman_param.x,
        "garch_rmse": calculate_rmse(returns[1:], garch_param.x[0]),
        "kalman_rmse": calculate_rmse(returns[1:], kalman_param.x[0]),
    }


#     return result
