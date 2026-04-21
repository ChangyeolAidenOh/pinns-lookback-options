import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# PINN 네트워크 정의
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

# 2차원 Lookback 옵션 Payoff 함수 정의
def double_lookback_payoff(S1, S2, K):
    return tf.maximum(S1 - S2 - K, 0.0)

# 손실 함수 정의
def compute_loss(model, S1, S2, K, r, sigma1, sigma2, rho, T):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(S1)
        tape.watch(S2)
        pred = model(tf.concat([S1, S2], axis=1))
        grad_S1 = tape.gradient(pred, S1)
        grad_S2 = tape.gradient(pred, S2)

    joint_density = tf.exp(-(S1**2 + S2**2 - 2 * rho * S1 * S2) / (2 * (1 - rho**2)))
    PDE_loss = tf.square(grad_S1 + grad_S2 + joint_density + double_lookback_payoff(S1, S2, K))
    return tf.reduce_mean(PDE_loss)

# 학습 데이터 생성
def generate_data(num_points=1000):
    S1 = tf.linspace(50.0, 150.0, num_points)[:, None]
    S2 = tf.linspace(50.0, 150.0, num_points)[:, None]
    return tf.Variable(S1), tf.Variable(S2)

# 학습 루프
def train_pinn(model, optimizer, epochs=100):
    S1, S2 = generate_data()
    K, r, sigma1, sigma2, rho, T = 10.0, 0.05, 0.2, 0.2, 0.5, 1.0
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, S1, S2, K, r, sigma1, sigma2, rho, T)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")

# 모델 초기화 및 학습
layers = [2, 50, 50, 1]  # Input: S1, S2
pinn_model = PINN(layers)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
train_pinn(pinn_model, optimizer, epochs=100)

# 테스트 및 시각화
S1_test = tf.linspace(50.0, 150.0, 100)[:, None]
S2_test = tf.linspace(50.0, 150.0, 100)[:, None]
predictions = pinn_model(tf.concat([S1_test, S2_test], axis=1))

plt.plot(S1_test.numpy(), predictions.numpy(), label="PINN Predicted Solution")
plt.xlabel("Stock Price (S1)")
plt.ylabel("Option Price")
plt.title("Double Lookback Option Value (PINN - TensorFlow)")
plt.legend()
plt.grid(True)
plt.show()
