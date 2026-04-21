import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

## Class definition
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

# Black-Scholes
def d1(S, K, T, r, sigma):
    T = torch.tensor(T, dtype=torch.float32)
    return (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * torch.sqrt(torch.tensor(T, dtype=torch.float32))

def black_scholes_lookback_call(S, M, r, T, sigma):

    T_tensor = torch.tensor(T, dtype=torch.float32)  # T를 Tensor로 변환
    r_tensor = torch.tensor(r, dtype=torch.float32)  # r를 Tensor로 변환

    d1_val = d1(S, M, T_tensor, r_tensor, sigma)
    d2_val = d2(S, M, T_tensor, r_tensor, sigma)
    cdf_d1 = 0.5 * (1 + torch.erf(d1_val / torch.sqrt(torch.tensor(2.0))))
    cdf_d2 = 0.5 * (1 + torch.erf(d2_val / torch.sqrt(torch.tensor(2.0))))

    return S * cdf_d1 - M * torch.exp(-r_tensor * T_tensor) * cdf_d2

# loss
def loss_function(model, S, M, r, sigma, T):
    S.requires_grad = True
    pred = model(S)
    d_pred = torch.autograd.grad(pred, S, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
    PDE_loss = (d_pred + black_scholes_lookback_call(S, M, r, T, sigma))**2
    return torch.mean(PDE_loss)

## Data set
def generate_data(num_points=1000):
    S = torch.linspace(50, 150, num_points).view(-1, 1).requires_grad_()
    return S

# Loop
def train_pinn(model, optimizer, epochs=100, lr=1e-3):
    S = generate_data()
    M, r, sigma, T = 120, 0.05, 0.2, 1.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model, S, M, r, sigma, T)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

## Initialized setting
layers = [1, 50, 50, 1]
pinn_model = PINN(layers)
optimizer = optim.Adam(pinn_model.parameters(), lr=1e-3)
train_pinn(pinn_model, optimizer, epochs=100)

## Test case
S_test = torch.linspace(50, 150, 100).view(-1, 1)
predictions = pinn_model(S_test).detach().numpy()

plt.plot(S_test.numpy(), predictions, label="PINN Predicted Solution")
plt.xlabel("Stock Price (S)")
plt.ylabel("Option Price")
plt.title("Lookback Option Value (PINN - PyTorch)")
plt.legend()
plt.grid(True)
plt.show()
