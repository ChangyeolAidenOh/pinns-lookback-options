import torch
import torch.nn as nn
import torch.optim as optim
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

# Green's Function 정의
def greens_function(xi, tau, x, sigma, r):
    # 텐서로 변환
    sigma = torch.tensor(sigma, dtype=torch.float32)
    r = torch.tensor(r, dtype=torch.float32)
    tau = torch.tensor(tau, dtype=torch.float32)
    alpha = r / sigma**2 - 0.5
    beta = -0.5 * (r / sigma**2 + 0.5)**2
    pi_tensor = torch.tensor(torch.pi, dtype=torch.float32)

    # psi 함수
    psi = lambda x, tau: (1 / torch.sqrt(2 * pi_tensor * sigma**2 * tau)) * torch.exp(-x**2 / (2 * sigma**2 * tau))

    return (psi(x - xi, tau) - psi(x + xi, tau)) * torch.exp(alpha * (x - xi) + beta * tau)

# 손실 함수 정의
def loss_function(model, x, xi, tau, sigma, r):
    x.requires_grad = True
    pred = model(x)
    d_pred = torch.autograd.grad(pred, x, grad_outputs=torch.ones_like(pred), create_graph=True)[0]

    # PDE 기반 손실
    greens_val = greens_function(xi, tau, x, sigma, r)
    PDE_loss = (d_pred + greens_val)**2
    return torch.mean(PDE_loss)

# 학습 데이터 생성
def generate_data(num_points=1000):
    x = torch.linspace(-3, 3, num_points).view(-1, 1).requires_grad_()
    return x

# 학습 루프
def train_pinn(model, optimizer, epochs=100, lr=1e-3):
    x = generate_data()
    xi, tau, sigma, r = 0.0, 1.0, 0.2, 0.05
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model, x, xi, tau, sigma, r)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 모델 초기화 및 학습
layers = [1, 50, 50, 1]
pinn_model = PINN(layers)
optimizer = optim.Adam(pinn_model.parameters(), lr=1e-3)
train_pinn(pinn_model, optimizer, epochs=100)

# 테스트 및 시각화
x_test = torch.linspace(-3, 3, 100).view(-1, 1)
predictions = pinn_model(x_test).detach().numpy()

plt.plot(x_test.numpy(), predictions, label="PINN Predicted Green's Function")
plt.xlabel("x")
plt.ylabel("Green's Function Value")
plt.title("Green's Function Approximation (PINN - PyTorch)")
plt.legend()
plt.grid(True)
plt.show()
