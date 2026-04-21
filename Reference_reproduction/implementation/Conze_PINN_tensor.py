import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_probability import distributions as tfp
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


# PINN 모델 정의
class PINN(tf.keras.Model):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(layer, activation='tanh') for layer in layers[:-1]]
        self.output_layer = tf.keras.layers.Dense(layers[-1], activation=None)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

# Black-Scholes 기반 도우미 함수
def d1(S, K, T, r, sigma):
    return (tf.math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * tf.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * tf.sqrt(T)

def black_scholes_lookback_call(S, M, r, T, sigma):
    """
    Standard lookback call option pricing using Black-Scholes.
    """
    d1_val = d1(S, M, T, r, sigma)
    d2_val = d2(S, M, T, r, sigma)
    cdf_d1 = tfp.Normal(loc=0.0, scale=1.0).cdf(d1_val)
    cdf_d2 = tfp.Normal(loc=0.0, scale=1.0).cdf(d2_val)
    return S * cdf_d1 - M * tf.exp(-r * T) * cdf_d2

# 손실 함수
@tf.function
def compute_loss(model, S, M, r, sigma, T):
    with tf.GradientTape() as tape:
        tape.watch(S)
        predictions = model(S)
    d_predictions = tape.gradient(predictions, S)
    loss = tf.reduce_mean(tf.square(d_predictions + black_scholes_lookback_call(S, M, r, T, sigma)))
    return loss

# 학습 루프
def train_pinn(model, epochs=1000, lr=1e-3):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    S = tf.convert_to_tensor(np.linspace(50, 150, 1000).reshape(-1, 1), dtype=tf.float32)
    M, r, sigma, T = 120, 0.05, 0.2, 1.0
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, S, M, r, sigma, T)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")

# 모델 초기화 및 학습
layers = [1, 50, 50, 1]
pinn_model = PINN(layers)
train_pinn(pinn_model, epochs=100, lr=1e-3)

# 테스트 및 시각화
S_test = tf.convert_to_tensor(np.linspace(50, 150, 100).reshape(-1, 1), dtype=tf.float32)
predictions = pinn_model(S_test).numpy()

plt.plot(S_test.numpy(), predictions, label="PINN Predicted Solution")
plt.xlabel("Stock Price (S)")
plt.ylabel("Option Price")
plt.title("Lookback Option Value (PINN - TensorFlow)")
plt.legend()
plt.grid(True)
plt.show()
