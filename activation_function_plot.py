import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Define the GELU activation function
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

# Define the continuous ternary activation function (using a simple approximation)
def continuous_ternary(x):
    return 1.5 * torch.tanh(x)+0.3*torch.tanh(-4*x)

# Generate a range of values from -3 to 3
x = torch.linspace(-3, 3, 100)

# Apply the ReLU, GELU, and continuous ternary activation functions
y_relu = F.relu(x)
y_gelu = gelu(x)
y_ternary = continuous_ternary(x)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), y_relu.numpy(), label='ReLU', linewidth=2)
plt.plot(x.numpy(), y_gelu.numpy(), label='GELU', linestyle='--', linewidth=2)
plt.plot(x.numpy(), y_ternary.numpy(), label='Continuous Ternary', linestyle='-.', linewidth=2)

plt.title('Comparison of Activation Functions')
plt.xlabel('Input value')
plt.ylabel('Activated value')
plt.legend()
plt.grid(True)
plt.savefig("act.png")
plt.show()

