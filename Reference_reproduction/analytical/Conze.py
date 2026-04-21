import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
S0 = 100  # Initial stock price
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility
T = 1  # Time to maturity (in years)
M = 120  # Maximum stock price (for extrema options)
Q = 80   # Minimum stock price (for extrema options)

# Utility functions for Black-Scholes components
def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

# Standard Lookback Call
def standard_lookback_call(S, m, r, T, sigma):
    """
    Compute the value of a standard lookback call option.
    """
    d1_val = d1(S, m, T, r, sigma)
    d2_val = d2(S, m, T, r, sigma)
    call_value = S * norm.cdf(d1_val) - m * np.exp(-r * T) * norm.cdf(d2_val)
    return call_value

# Standard Lookback Put
def standard_lookback_put(S, M, r, T, sigma):
    """
    Compute the value of a standard lookback put option.
    """
    d1_val = d1(S, M, T, r, sigma)
    d2_val = d2(S, M, T, r, sigma)
    put_value = M * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)
    return put_value

# Extrema Options
def option_on_extrema_call(S, M, r, T, sigma):
    """
    Compute the value of a call option on the maximum stock price.
    """
    d1_val = d1(S, M, T, r, sigma)
    d2_val = d2(S, M, T, r, sigma)
    return S * norm.cdf(d1_val) - M * np.exp(-r * T) * norm.cdf(d2_val)

def option_on_extrema_put(S, Q, r, T, sigma):
    """
    Compute the value of a put option on the minimum stock price.
    """
    d1_val = d1(S, Q, T, r, sigma)
    d2_val = d2(S, Q, T, r, sigma)
    return Q * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)

# Greeks: Delta, Gamma, Theta
def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Compute Delta, Gamma, and Theta for a given option type.
    """
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    delta = norm.cdf(d1_val) if option_type == "call" else -norm.cdf(-d1_val)
    gamma = norm.pdf(d1_val) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1_val) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2_val if option_type == "call" else -d2_val))
    return delta, gamma, theta

# Example computations
call_value = standard_lookback_call(S0, Q, r, T, sigma)
put_value = standard_lookback_put(S0, M, r, T, sigma)
extrema_call = option_on_extrema_call(S0, M, r, T, sigma)
extrema_put = option_on_extrema_put(S0, Q, r, T, sigma)

# Greeks for Lookback Call
delta, gamma, theta = calculate_greeks(S0, Q, T, r, sigma, option_type="call")

# Report results
print(f"Standard Lookback Call: {call_value:.2f}")
print(f"Standard Lookback Put: {put_value:.2f}")
print(f"Option on Extrema Call: {extrema_call:.2f}")
print(f"Option on Extrema Put: {extrema_put:.2f}")
print("\nGreeks:")
print(f"Delta (Call): {delta:.2f}")
print(f"Gamma: {gamma:.6f}")
print(f"Theta (Call): {theta:.2f}")

# Visualization
S_range = np.linspace(60, 140, 100)
call_values = [standard_lookback_call(S, Q, r, T, sigma) for S in S_range]
put_values = [standard_lookback_put(S, M, r, T, sigma) for S in S_range]

plt.figure(figsize=(10, 6))
plt.plot(S_range, call_values, label="Standard Lookback Call", color="blue")
plt.plot(S_range, put_values, label="Standard Lookback Put", color="red")
plt.title("Lookback Option Value vs. Stock Price")
plt.xlabel("Stock Price (S)")
plt.ylabel("Option Value")
plt.legend()
plt.grid(True)
plt.show()


