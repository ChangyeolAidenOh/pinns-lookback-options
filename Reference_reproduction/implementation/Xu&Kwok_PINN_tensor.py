import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class PINN(tf.keras.Model):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.hidden_layers = []
        for i in range(len(layers) - 2):
            self.hidden_layers.append(tf.keras.layers.Dense(layers[i + 1], activation='tanh'))
        self.output_layer = tf.keras.layers.Dense(layers[-1])

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


def greens_function(xi, tau, x, sigma, r):
    alpha = r / sigma**2 - 0.5
    beta = -0.5 * (r / sigma**2 + 0.5)**2
    psi = lambda x, tau: (1 / tf.sqrt(2 * np.pi * sigma**2 * tau)) * tf.exp(-x**2 / (2 * sigma**2 * tau))
    return (psi(x - xi, tau) - psi(x + xi, tau)) * tf.exp(alpha * (x - xi) + beta * tau)


def compute_loss(model, x, xi, tau, sigma, r):
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model(x)
        d_pred = tape.gradient(pred, x)

    greens_val = greens_function(xi, tau, x, sigma, r)
    PDE_loss = tf.square(d_pred + greens_val)
    return tf.reduce_mean(PDE_loss)


def generate_data(num_points=1000):
    x = tf.linspace(-3.0, 3.0, num_points)[:, None]
    return tf.Variable(x)


def train_pinn(model, optimizer, epochs=100, lr=1e-3):
    x = generate_data()
    xi, tau, sigma, r = 0.0, 1.0, 0.2, 0.05
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, x, xi, tau, sigma, r)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")

## Example case
layers = [1, 50, 50, 1]
pinn_model = PINN(layers)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
train_pinn(pinn_model, optimizer, epochs=100)

## Plot_visualization
x_test = tf.linspace(-3.0, 3.0, 100)[:, None]
predictions = pinn_model(x_test)

plt.plot(x_test.numpy(), predictions.numpy(), label="PINN Predicted Green's Function")
plt.xlabel("x")
plt.ylabel("Green's Function Value")
plt.title("Green's Function Approximation (PINN - TensorFlow)")
plt.legend()
plt.grid(True)
plt.show()
