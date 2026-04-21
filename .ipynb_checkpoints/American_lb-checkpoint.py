import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
'''
# Set data type
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

# Define parameters
T = tf.constant(2.0, dtype=DTYPE)  # Maturity time
r = tf.constant(0.05, dtype=DTYPE)  # Interest rate
sigma = tf.constant(0.2, dtype=DTYPE)  # Volatility
K = tf.constant(100.0, dtype=DTYPE)  # Strike price
a = tf.constant(75.0, dtype=DTYPE)  # Lower bound for asset price
b = tf.constant(125.0, dtype=DTYPE)  # Upper bound for asset price

# Payoff functions
def fixed_call_payoff(S_paths):
    min_S = tf.reduce_min(S_paths, axis=1, keepdims=True)  # Minimum asset price
    payoff = tf.maximum(S_paths[:, -1:] - min_S, 0.0)
    return tf.tile(payoff, [1, S_paths.shape[1]])  # Match time_steps dimension

def fixed_put_payoff(S_paths):
    min_S = tf.reduce_min(S_paths, axis=1, keepdims=True)
    payoff = tf.maximum(min_S - S_paths[:, -1:], 0.0)
    return tf.tile(payoff, [1, S_paths.shape[1]])  # Match time_steps dimension

# Generate paths
def generate_paths(batch_size, time_steps):
    S0 = tf.random.uniform((batch_size, 1), minval=a, maxval=b, dtype=DTYPE)  # Initial asset price
    dt = T / time_steps
    paths = [S0]
    for _ in range(1, time_steps):
        dW = tf.random.normal((batch_size, 1), mean=0.0, stddev=tf.sqrt(dt), dtype=DTYPE)
        St = paths[-1] * tf.exp((r - 0.5 * sigma**2) * dt + sigma * dW)
        paths.append(St)
    return tf.concat(paths, axis=1)  # Shape: (batch_size, time_steps)

# Define model
def init_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(50, activation='tanh'),
        tf.keras.layers.Dense(50, activation='tanh'),
        tf.keras.layers.Dense(50, activation='tanh'),
        tf.keras.layers.Dense(1, activation=None)
    ])
    return model

# PDE and loss
def pde_loss(tape, S, t, u, u_S):
    u_t = tape.gradient(u, t)
    u_SS = tape.gradient(u_S, S)
    return u_t + 0.5 * sigma**2 * S**2 * u_SS + r * S * u_S - r * u

@tf.function
def loss_function(model, S, t, payoff):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(S)
        tape.watch(t)
        inputs = tf.concat([S, t], axis=1)  # Combine S and t into a single input
        u = model(inputs)  # Model output
        u_S = tape.gradient(u, S)
        loss_pde = tf.reduce_mean(tf.square(pde_loss(tape, S, t, u, u_S)))  # PDE loss
    payoff = tf.reshape(payoff, (-1, 1))  # Reshape payoff to match (batch_size * time_steps, 1)
    loss_payoff = tf.reduce_mean(tf.square(u - payoff))  # Payoff loss
    return loss_pde + loss_payoff

# Training loop
def train_pinn(model, payoff_function, epochs=500, batch_size=1000, time_steps=100):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    for epoch in range(epochs):
        S_paths = generate_paths(batch_size, time_steps)  # Generate paths
        S = tf.reshape(S_paths, (-1, 1))  # Flatten S_paths for input to the model
        t = tf.linspace(0.0, T, time_steps)  # Generate time steps
        t = tf.tile(t[tf.newaxis, :], [batch_size, 1])
        t = tf.reshape(t, (-1, 1))  # Flatten time steps
        payoff = payoff_function(S_paths)  # Compute payoff
        with tf.GradientTape() as tape:
            loss = loss_function(model, S, t, payoff)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")

# Monte Carlo simulation
def monte_carlo_price(payoff_function, simulations=100000, time_steps=100):
    S_paths = generate_paths(simulations, time_steps)
    payoff = payoff_function(S_paths)
    return tf.reduce_mean(payoff[:, -1]).numpy()

# Plotting
def plot_option_value(model, payoff_function, option_name, batch_size=1000, time_steps=100):
    S_paths = generate_paths(batch_size, time_steps)
    S = tf.reshape(S_paths[:, -1:], (-1, 1))  # Use terminal prices
    t = tf.zeros_like(S)  # Time is zero for terminal payoff
    inputs = tf.concat([S, t], axis=1)
    predictions = model(inputs)
    monte_carlo = monte_carlo_price(payoff_function)
    plt.figure(figsize=(8, 6))
    plt.scatter(S.numpy(), predictions.numpy(), alpha=0.5, label=f"{option_name} (PINN Prediction)")
    plt.axhline(monte_carlo, color='red', linestyle='--', label=f"Monte Carlo Price: {monte_carlo:.2f}")
    plt.xlabel("Stock Price (S)")
    plt.ylabel("Option Value")
    plt.title(f"{option_name} Option Value")
    plt.legend()
    plt.grid()
    plt.show()

# Main
option_types = [
    ("Fixed Call", fixed_call_payoff),
    ("Fixed Put", fixed_put_payoff),
]

for option_name, payoff_function in option_types:
    print(f"Training for {option_name}")
    model = init_model()
    train_pinn(model, payoff_function)
    print(f"Monte Carlo Price for {option_name}: {monte_carlo_price(payoff_function):.2f}")
    plot_option_value(model, payoff_function, option_name)

'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Data type
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

# Parameters
T = tf.constant(1.0, dtype=DTYPE)  # Maturity
r = tf.constant(0.05, dtype=DTYPE)  # Risk-free rate
sigma = tf.constant(0.2, dtype=DTYPE)  # Volatility
K = tf.constant(100.0, dtype=DTYPE)  # Strike price

# Monte Carlo simulation for option pricing
def monte_carlo_simulation(option_type='fixed_call', num_simulations=10000, num_steps=100):
    S0 = tf.random.uniform(shape=(num_simulations, 1), minval=50, maxval=150, dtype=DTYPE)
    dt = T / num_steps
    S_paths = [S0]

    for _ in range(num_steps - 1):
        dW = tf.random.normal(shape=(num_simulations, 1), mean=0.0, stddev=tf.sqrt(dt), dtype=DTYPE)
        S_next = S_paths[-1] * tf.exp((r - 0.5 * sigma**2) * dt + sigma * dW)
        S_paths.append(S_next)

    S_paths = tf.concat(S_paths, axis=1)
    min_S = tf.reduce_min(S_paths, axis=1, keepdims=True)
    S_T = S_paths[:, -1:]
    if option_type == 'fixed_call':
        payoff = tf.maximum(S_T - min_S, 0.0)
    elif option_type == 'fixed_put':
        payoff = tf.maximum(K - min_S, 0.0)
    else:
        raise ValueError("Invalid option type")

    option_price = tf.reduce_mean(payoff) * tf.exp(-r * T)
    return option_price.numpy(), S_paths

# Test Monte Carlo simulation
fixed_call_price, fixed_call_paths = monte_carlo_simulation(option_type='fixed_call')
fixed_put_price, fixed_put_paths = monte_carlo_simulation(option_type='fixed_put')
#print(fixed_call_paths[0].shape)
#print(len(fixed_call_price))
# Visualize price paths
plt.figure(figsize=(10, 6))
for i in np.arange(30,40):  # Plot 10 paths
    plt.plot(fixed_call_paths[i], lw=0.5)
    #print(fixed_call_price.shape)
plt.title('Monte Carlo Simulated Price Paths')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.grid()
plt.show()

# Display Monte Carlo prices
print(f"Monte Carlo Fixed Call Price: {fixed_call_price}")
print(f"Monte Carlo Fixed Put Price: {fixed_put_price}")


""""""""""""""""
import nbformat
def save_as_jupyter(py_file_path, ipynb_file_path):
    
    with open(py_file_path, 'r') as py_file:
        py_content = py_file.read()

    
    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(nbformat.v4.new_code_cell(py_content))

    
    with open(ipynb_file_path, 'w') as ipynb_file:
        nbformat.write(notebook, ipynb_file)



save_as_jupyter('example.py', 'output.ipynb')
"""""""""""""""