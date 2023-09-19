# PyTorch Basics Tutorial

PyTorch is an open-source deep learning library that provides a flexible and dynamic framework for building neural networks. In this tutorial, we'll cover the following topics:

1. **Tensors**: The fundamental data structure in PyTorch.
2. **Operations**: Basic tensor operations.
3. **Autograd**: Automatic differentiation in PyTorch.
4. **Simple Neural Network**: Creating and training a basic neural network.

## Installation

You can install PyTorch using pip. Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for installation instructions specific to your system.

## Importing PyTorch

To start using PyTorch, import the library in your Python script:

```python
import torch
```
### Creating Tensors
```python
# Create a tensor from a Python list
a = torch.tensor([1, 2, 3])
print(a)

# Create a tensor filled with zeros
b = torch.zeros(2, 3)
print(b)

# Create a tensor with random values
c = torch.rand(3, 4)
print(c)
```
#### Tensor Operations
```python
# Element-wise addition
result = a + b
print(result)

# Matrix multiplication
result = torch.matmul(b, c)
print(result)

# Reshaping a tensor
d = torch.arange(9).reshape(3, 3)
print(d)
```
#### Autograd
```python
# Create a tensor with requires_grad=True to track operations for gradient computation
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x + 1

# Compute gradients
y.backward()
print(x.grad)  # Access the gradient
```
##### Simple neural networks
```python
import torch.nn as nn
import torch.optim as optim

# Define a simple feedforward neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        return x

# Create a model
model = Net()

# Define a loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop (assuming you have data)
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```
