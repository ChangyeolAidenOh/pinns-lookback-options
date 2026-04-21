import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# PINN 네트워크 정의
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                layer_list.append(nn.Tanh())
        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)

# 2차원 Lookback 옵션 Payoff 함수 정의
def double_lookback_payoff(S1, S2, K):
    """
    Payoff for a double lookback call option.
    """
    return torch.maximum(S1 - S2 - K, torch.tensor(0.0, dtype=torch.float32))

# 손실 함수 정의
def loss_function(model, S1, S2, K, r, sigma1, sigma2, rho, T):
    S1.requires_grad = True
    S2.requires_grad = True
    pred = model(torch.cat((S1, S2), dim=1))
    grad_S1 = torch.autograd.grad(pred, S1, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
    grad_S2 = torch.autograd.grad(pred, S2, grad_outputs=torch.ones_like(pred), create_graph=True)[0]

    # Joint density approximation (using simplified form for PINN training)
    joint_density = torch.exp(-(S1 ** 2 + S2 ** 2 - 2 * rho * S1 * S2) / (2 * (1 - rho ** 2)))

    PDE_loss = (grad_S1 + grad_S2 + joint_density + double_lookback_payoff(S1, S2, K)) ** 2
    return torch.mean(PDE_loss)

# 학습 데이터 생성
def generate_data(num_points=1000):
    S1 = torch.linspace(50, 150, num_points).view(-1, 1).requires_grad_()
    S2 = torch.linspace(50, 150, num_points).view(-1, 1).requires_grad_()
    return S1, S2

# 학습 루프
def train_pinn(model, optimizer, epochs=100):
    S1, S2 = generate_data()
    K, r, sigma1, sigma2, rho, T = 10, 0.05, 0.2, 0.2, 0.5, 1.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model, S1, S2, K, r, sigma1, sigma2, rho, T)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 모델 초기화 및 학습
layers = [2, 50, 50, 1]  # Input: S1, S2
pinn_model = PINN(layers)
optimizer = optim.Adam(pinn_model.parameters(), lr=1e-3)
train_pinn(pinn_model, optimizer, epochs=100)

# 테스트 및 시각화
S1_test = torch.linspace(50, 150, 100).view(-1, 1)
S2_test = torch.linspace(50, 150, 100).view(-1, 1)
predictions = pinn_model(torch.cat((S1_test, S2_test), dim=1)).detach().numpy()

plt.plot(S1_test.numpy(), predictions, label="PINN Predicted Solution")
plt.xlabel("Stock Price (S1)")
plt.ylabel("Option Price")
plt.title("Double Lookback Option Value (PINN - PyTorch)")
plt.legend()
plt.grid(True)
plt.show()
