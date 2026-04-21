import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt

# Constants
S1 = 100  # Initial price of asset 1
S2 = 100  # Initial price of asset 2
r = 0.05  # Risk-free interest rate
T = 1.0  # Time to maturity


# Safe exponential function to avoid overflow
def safe_exp(x):
    return np.exp(np.minimum(x, 700))  # Cap exponentials to avoid overflow


# Payoff function for double lookback call option
def double_lookback_payoff(x1, x2, strike):
    """
    Payoff for a double lookback call option based on the difference
    between the maxima of two assets.
    """
    return max(S1 * safe_exp(x1) - S2 * safe_exp(x2) - strike, 0)


# Joint density function of two correlated Brownian motions
def joint_density(x1, x2, sigma1, sigma2, rho, T):
    """
    Joint probability density function of two correlated Brownian motions
    at time T.
    """
    sigma1_T = sigma1 * np.sqrt(T)
    sigma2_T = sigma2 * np.sqrt(T)
    denom = 2 * np.pi * sigma1_T * sigma2_T * np.sqrt(1 - rho ** 2)

    z1 = x1 / sigma1_T
    z2 = x2 / sigma2_T
    exp_term = -((z1 ** 2 + z2 ** 2 - 2 * rho * z1 * z2) / (2 * (1 - rho ** 2)))

    return np.exp(exp_term) / denom


# Numerical integration with bounded range
def calculate_option_price(payoff_function, sigma1, sigma2, rho, T, strike):
    """
    Calculate the price of a lookback option by integrating the payoff
    function with respect to the joint density of maxima and minima.
    """
    result, error = dblquad(
        lambda x1, x2: payoff_function(x1, x2, strike) * joint_density(x1, x2, sigma1, sigma2, rho, T),
        -5, 5,  # Limit x1 bounds
        lambda x1: -5, lambda x1: 5  # Limit x2 bounds
    )
    return np.exp(-r * T) * result  # Discounted present value of expected payoff


# Plotting the joint density for visualization
def plot_joint_density(sigma1, sigma2, rho, T):
    """
    Plot the joint density function for visualization.
    """
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = joint_density(X, Y, sigma1, sigma2, rho, T)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title("Semilookback Joint Density")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Density")
    plt.show()


# Parameters
sigma1 = 0.2  # Volatility of asset 1
sigma2 = 0.2  # Volatility of asset 2
rho = 0.5  # Correlation between assets

# Strike prices for validation
strikes = [90, 100, 110]

# Validate the option pricing
print("Validating Lookback Options")
for strike in strikes:
    try:
        double_price = calculate_option_price(double_lookback_payoff, sigma1, sigma2, rho, T, strike)
        print(f"Strike: {strike}, Double Lookback Call: {double_price:.4f}")
    except Exception as e:
        print(f"Strike: {strike}, Error: {e}")

# Plot the joint density
plot_joint_density(sigma1, sigma2, rho, T)
