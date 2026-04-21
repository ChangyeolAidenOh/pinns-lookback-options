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

# Define the lookback payoff function
def lookback_payoff(S_paths):
    min_S = tf.reduce_min(S_paths, axis=1, keepdims=True)  # Minimum asset price (batch_size, 1)
    terminal_S = S_paths[:, -1:]  # Final asset price (batch_size, 1)
    payoff = tf.maximum(terminal_S - min_S, 0.0)  # Calculate the payoff for the lookback option
    return tf.tile(payoff, [1, S_paths.shape[1]])  # Replicate payoff for time steps (batch_size, time_steps)

# Initialize the model
def init_model(num_hidden_neurons=50, num_hidden_layers=3):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(2,)))  # Input dimension (S, t)
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_hidden_neurons, activation='tanh'))  # Hidden layers with tanh activation
    model.add(tf.keras.layers.Dense(1, activation=None))  # Output layer
    return model

# Define the PDE (partial differential equation)
def pde(tape, S, t, u, u_S):
    u_t = tape.gradient(u, t)  # Partial derivative with respect to time
    u_SS = tape.gradient(u_S, S)  # Second-order partial derivative with respect to S
    pde_loss = u_t + 0.5 * sigma ** 2 * S ** 2 * u_SS + r * S * u_S - r * u  # PDE formulation
    return pde_loss

# Define the loss function
@tf.function
def loss_function(model, S, t, payoff, epoch=None):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(S)
        tape.watch(t)

        # Combine inputs (S and t) for the model
        combined_input = tf.concat([S, t], axis=1)
        if epoch is not None and epoch % 100 == 0:
            tf.print("Epoch", epoch, "- Shape of combined_input:", tf.shape(combined_input))  # Debugging output

        u = model(combined_input)  # Model prediction
        u_S = tape.gradient(u, S)  # First-order derivative with respect to S
        pde_loss = pde(tape, S, t, u, u_S)  # Calculate PDE loss

    payoff_loss = tf.reduce_mean(tf.square(u - payoff))  # Payoff loss
    return tf.reduce_mean(tf.square(pde_loss)) + payoff_loss  # Total loss

# Generate asset price paths
def generate_paths(batch_size, time_steps):
    S0 = tf.random.uniform((batch_size, 1), minval=75, maxval=125, dtype=DTYPE)  # Initial asset price
    dt = T / time_steps  # Time step size
    S_paths = [S0]  # Initialize paths with the initial price

    for _ in range(1, time_steps):
        dW = tf.random.normal((batch_size, 1), mean=0.0, stddev=tf.sqrt(dt), dtype=DTYPE)  # Brownian motion
        St = S_paths[-1] * tf.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)  # Simulate next price step
        S_paths.append(St)

    return tf.concat(S_paths, axis=1)  # Concatenate all price steps (batch_size, time_steps)

# Training loop
def train_pinn(model, optimizer, epochs=500, batch_size=1000, time_steps=100):
    for epoch in range(epochs):
        S_paths = generate_paths(batch_size, time_steps)  # Generate simulated paths

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, S_paths shape: {S_paths.shape}")  # Debugging output

        S = tf.reshape(S_paths, (batch_size * time_steps, 1))  # Reshape for model input
        t = tf.linspace(0.0, T, time_steps)  # Time vector
        t = tf.cast(t, dtype=DTYPE)  # Cast to correct dtype
        t = tf.tile(t[tf.newaxis, :], [batch_size, 1])
        t = tf.reshape(t, (batch_size * time_steps, 1))  # Reshape for model input

        payoff = lookback_payoff(S_paths)  # Compute payoff

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, payoff shape: {payoff.shape}")  # Debugging output

        payoff = tf.reshape(payoff, (batch_size * time_steps, 1))  # Reshape for model output

        with tf.GradientTape() as tape:
            loss = loss_function(model, S, t, payoff)  # Compute loss
        grads = tape.gradient(loss, model.trainable_variables)  # Calculate gradients
        optimizer.apply_gradients(zip(grads, model.trainable_variables))  # Apply gradients

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")  # Print loss periodically

# Visualize the solution
def plot_solution(model, batch_size=1000, time_steps=100):
    S_paths = generate_paths(batch_size, time_steps)  # Generate test paths
    S = S_paths[:, 0]  # Extract the first time step (initial prices)

    t = tf.linspace(0.0, T, time_steps)  # Time vector
    t = tf.cast(t, dtype=DTYPE)  # Cast to correct dtype
    t = tf.tile(t[tf.newaxis, :], [batch_size, 1])
    t = tf.reshape(t, (batch_size * time_steps, 1))

    S = tf.reshape(S, (batch_size, 1))  # Reshape for plotting
    combined_input = tf.concat([S, t[:batch_size]], axis=1)  # Combine inputs for prediction
    predictions = model.predict(combined_input)  # Get predictions

    plt.figure(figsize=(10, 6))
    plt.plot(S.numpy(), predictions, label="PINN Predicted Solution")
    plt.xlabel("Stock Price (S)")
    plt.ylabel("Option Value")
    plt.title("Lookback Option Value (PINN - TensorFlow)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Initialize and train the model
pinn_model = init_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

train_pinn(pinn_model, optimizer, epochs=500, batch_size=1000, time_steps=100)
plot_solution(pinn_model)
