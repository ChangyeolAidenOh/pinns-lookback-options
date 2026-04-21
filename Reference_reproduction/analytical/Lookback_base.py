import numpy as np
from scipy.stats import norm

# Define parameters
def lookback_option_params():
    """
    Define the parameters for lookback options.
    Returns:
        dict: A dictionary containing model parameters.
    """
    return {
        "S0": 100,  # Initial stock price
        "M0": 120,  # Maximum stock price
        "m0": 80,   # Minimum stock price
        "K": 110,   # Strike price
        "T": 1,     # Time to maturity
        "r": 0.05,  # Risk-free rate
        "sigma": 0.2  # Volatility
    }

# Compute d values for the equations
def compute_d(params):
    """
    Compute d0, d1, and d2 based on the parameters.

    Args:
        params (dict): Parameters for the lookback options.

    Returns:
        dict: A dictionary containing d values.
    """
    S0, M0, m0, K, T, r, sigma = params.values()
    d0 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d1 = d0 + np.log(K / M0) / (sigma * np.sqrt(T))
    d2 = d0 + np.log(K / m0) / (sigma * np.sqrt(T))
    return {"d0": d0, "d1": d1, "d2": d2}

# Floating strike call price
def floating_strike_call_price(params, d_values):
    """
    Calculate the price of a floating strike call lookback option.

    Args:
        params (dict): Parameters for the lookback options.
        d_values (dict): Pre-computed d values.

    Returns:
        float: The floating strike call price.
    """
    S0, M0, m0, T, r, sigma = params["S0"], params["M0"], params["m0"], params["T"], params["r"], params["sigma"]
    d2 = d_values["d2"]
    term1 = S0 * norm.cdf(d2)
    term2 = np.exp(-r * T) * m0 * norm.cdf(d2 - sigma * np.sqrt(T))
    term3 = (sigma ** 2) / (2 * r) * np.exp(-r * T) * S0 * (
        (S0 / m0) ** (-2 * r / sigma ** 2) * norm.cdf(-d2 + 2 * r / sigma * np.sqrt(T)) -
        np.exp(r * T) * norm.cdf(-d2)
    )
    return term1 - term2 + term3

# Fixed strike call price
def fixed_strike_call_price(params, d_values):
    """
    Calculate the price of a fixed strike call lookback option.

    Args:
        params (dict): Parameters for the lookback options.
        d_values (dict): Pre-computed d values.

    Returns:
        float: The fixed strike call price.
    """
    S0, M0, K, T, r, sigma = params["S0"], params["M0"], params["K"], params["T"], params["r"], params["sigma"]
    d0 = d_values["d0"]
    term1 = S0 * norm.cdf(d0)
    term2 = np.exp(-r * T) * K * norm.cdf(d0 - sigma * np.sqrt(T))
    term3 = (sigma ** 2) / (2 * r) * np.exp(-r * T) * S0 * (
        - (S0 / K) ** (-2 * r / sigma ** 2) * norm.cdf(d0 - 2 * r / sigma * np.sqrt(T)) +
        np.exp(r * T) * norm.cdf(d0)
    )
    return term1 - term2 + term3

# Example usage
params = lookback_option_params()
d_values = compute_d(params)
floating_call_price = floating_strike_call_price(params, d_values)
fixed_call_price = fixed_strike_call_price(params, d_values)

print(f"Floating Strike Call Price: {floating_call_price:.4f}")
print(f"Fixed Strike Call Price: {fixed_call_price:.4f}")
