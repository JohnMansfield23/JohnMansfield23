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
### Tensors
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
