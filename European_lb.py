import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set data type
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

# Define parameters
T = tf.constant(2.0, dtype=DTYPE)  # Maturity time
r = tf.constant(0.05, dtype=DTYPE)  # Interest rate
sigma = tf.constant(0.2, dtype=DTYPE)  # Volatility
mu = tf.constant(0.05 - 0.5 * 0.2**2, dtype=DTYPE)  # Drift: r - 0.5 * sigma^2

# Lookback option parameters
K = tf.constant(100.0, dtype=DTYPE)  # Strike price
a = tf.constant(75.0, dtype=DTYPE)  # Lower bound for asset price
b = tf.constant(125.0, dtype=DTYPE)  # Upper bound for asset price

# Lookback payoff functions
def floating_call_payoff(S_paths):
    min_S = tf.reduce_min(S_paths, axis=1, keepdims=True)  # Minimum price
    terminal_S = S_paths[:, -1:]  # Final price
    return tf.maximum(terminal_S - min_S, 0.0)

def floating_put_payoff(S_paths):
    max_S = tf.reduce_max(S_paths, axis=1, keepdims=True)  # Maximum price
    terminal_S = S_paths[:, -1:]  # Final price
    return tf.maximum(max_S - terminal_S, 0.0)

def fixed_call_payoff(S_paths):
    max_S = tf.reduce_max(S_paths, axis=1, keepdims=True)  # Maximum price
    return tf.maximum(max_S - K, 0.0)

def fixed_put_payoff(S_paths):
    min_S = tf.reduce_min(S_paths, axis=1, keepdims=True)  # Minimum price
    return tf.maximum(K - min_S, 0.0)

# Generate asset price paths
def generate_paths(batch_size, time_steps):
    S0 = tf.random.uniform((batch_size, 1), minval=a, maxval=b, dtype=DTYPE)  # Initial asset price
    dt = T / time_steps  # Time step size
    S_paths = [S0]

    for _ in range(1, time_steps):
        dW = tf.random.normal((batch_size, 1), mean=0.0, stddev=tf.sqrt(dt), dtype=DTYPE)  # Brownian motion
        St = S_paths[-1] * tf.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)  # Simulate next price step
        S_paths.append(St)

    return tf.concat(S_paths, axis=1)  # Concatenate all price steps (batch_size, time_steps)

# Training loop
def train_pinn(model, optimizer, payoff_function, epochs=500, batch_size=1000, time_steps=100):
    for epoch in range(epochs):
        S_paths = generate_paths(batch_size, time_steps)  # Generate simulated paths
        S = tf.reshape(S_paths, (batch_size * time_steps, 1))  # Reshape for model input
        t = tf.linspace(0.0, T, time_steps)  # Time vector
        t = tf.cast(t, dtype=DTYPE)  # Cast to correct dtype
        t = tf.tile(t[tf.newaxis, :], [batch_size, 1])
        t = tf.reshape(t, (batch_size * time_steps, 1))  # Reshape for model input

        payoff = payoff_function(S_paths)  # Compute payoff
        payoff = tf.reshape(payoff, (batch_size * time_steps, 1))  # Reshape for model output

        with tf.GradientTape() as tape:
            loss = loss_function(model, S, t, payoff)
        grads = tape.gradient(loss, model.trainable_variables)  # Calculate gradients
        optimizer.apply_gradients(zip(grads, model.trainable_variables))  # Apply gradients

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")

# Define the loss function
@tf.function
def loss_function(model, S, t, payoff):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(S)
        tape.watch(t)

        combined_input = tf.concat([S, t], axis=1)  # Combine inputs
        u = model(combined_input)  # Model prediction
        u_S = tape.gradient(u, S)  # First-order derivative with respect to S
        u_t = tape.gradient(u, t)  # First-order derivative with respect to t
        u_SS = tape.gradient(u_S, S)  # Second-order derivative with respect to S
        pde_loss = u_t + 0.5 * sigma**2 * S**2 * u_SS + r * S * u_S - r * u  # PDE

    payoff_loss = tf.reduce_mean(tf.square(u - payoff))  # Payoff loss
    return tf.reduce_mean(tf.square(pde_loss)) + payoff_loss  # Total loss

# Initialize the model
def init_model(num_hidden_neurons=50, num_hidden_layers=3):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(2,)))  # Input: (S, t)
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_hidden_neurons, activation='tanh'))  # Hidden layers
    model.add(tf.keras.layers.Dense(1, activation=None))  # Output layer
    return model

# Initialize model and optimizer
pinn_model = init_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Example: Train for Floating Call
print("Training for European Floating Call")
train_pinn(pinn_model, optimizer, floating_call_payoff)

# Example: Train for Fixed Put
print("Training for European Fixed Put")
train_pinn(pinn_model, optimizer, fixed_put_payoff)
