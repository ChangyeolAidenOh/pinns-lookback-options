import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt


# Define the Green's function based on the paper
def greens_function(xi, tau, x, sigma, r):
    """
    Compute the Green's function G(xi, tau; x) for the semi-infinite domain.
    Args:
        xi (float): Integration variable.
        tau (float): Time to maturity.
        x (float): Logarithmic variable ln(M/S).
        sigma (float): Volatility of the underlying asset.
        r (float): Risk-free interest rate.
    Returns:
        float: Value of the Green's function.
    """
    alpha = r / sigma ** 2 - 0.5
    beta = -0.5 * (r / sigma ** 2 + 0.5) ** 2
    psi = lambda x, tau: (1 / np.sqrt(2 * np.pi * sigma ** 2 * tau)) * np.exp(-x ** 2 / (2 * sigma ** 2 * tau))
    return (psi(x - xi, tau) - psi(x + xi, tau)) * np.exp(alpha * (x - xi) + beta * tau)

# Define the payoff function
def payoff_fixed_strike(M, K):
    """
    Payoff for the fixed strike lookback call option.
    Args:
        M (float): Maximum price of the underlying during the lookback period.
        K (float): Strike price.
    Returns:
        float: Payoff value.
    """
    return max(M - K, 0)


# Define the integral formula for lookback option pricing
def lookback_option_price(S, M, T, sigma, r, K, payoff_func):
    """
    Compute the price of a lookback option using the integral representation.
    Args:
        S (float): Current stock price.
        M (float): Maximum price during the lookback period.
        T (float): Time to maturity.
        sigma (float): Volatility of the underlying asset.
        r (float): Risk-free interest rate.
        K (float): Strike price for fixed strike options.
        payoff_func (function): Payoff function for the option.
    Returns:
        float: Price of the lookback option.
    """
    x = np.log(M / S)
    VT = lambda xi: payoff_func(M * np.exp(-xi), K)

    # First integral term
    def integrand_1(xi):
        return greens_function(xi, T, x, sigma, r) * VT(xi)

    # Compute the integral using scipy.quad
    integral_1, error_1 = quad(integrand_1, 0, np.inf)

    return integral_1 * np.exp(-r * T)


# Parameters
S = 100  # Current stock price
M = 120  # Maximum stock price during lookback period
T = 1  # Time to maturity (in years)
sigma = 0.2  # Volatility
r = 0.05  # Risk-free interest rate
K = 110  # Strike price for fixed strike option

# Compute the lookback option price
price_fixed = lookback_option_price(S, M, T, sigma, r, K, payoff_fixed_strike)
print(f"Fixed Strike Lookback Option Price: {price_fixed:.4f}")

# Plot Green's function for visualization
x_vals = np.linspace(-3, 3, 100)
tau_vals = np.linspace(0.01, 1, 50)
X, Tau = np.meshgrid(x_vals, tau_vals)
G_vals = np.vectorize(lambda x, tau: greens_function(0, tau, x, sigma, r))(X, Tau)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Tau, G_vals, cmap='viridis')
ax.set_title("Green's Function for Lookback Option")
ax.set_xlabel("Log Price (x)")
ax.set_ylabel("Time to Maturity (tau)")
ax.set_zlabel("G(xi, tau; x)")
plt.show()

# Define payoff for floating strike
def payoff_floating_strike(M, S):
    """
    Payoff for the floating strike lookback call option.
    Args:
        M (float): Maximum price of the underlying during the lookback period.
        S (float): Current stock price.
    Returns:
        float: Payoff value.
    """
    return max(M - S, 0)

# Compute the floating strike lookback option price
price_floating = lookback_option_price(S, M, T, sigma, r, K, lambda M, K: payoff_floating_strike(M, S))

# Check the parity relation
parity_relation = price_fixed - (price_floating + S - K * np.exp(-r * T))
print(f"Floating Strike Lookback Option Price: {price_floating:.4f}")
print(f"Parity Relation Difference: {parity_relation:.4e}")

# Optional: Include any additional visualizations or checks if needed