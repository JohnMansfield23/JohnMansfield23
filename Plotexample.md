# ploting Y=mx+b in python
```python
import matplotlib.pyplot as plt
import numpy as np

# Define the values for m and b
m = 2
b = 1

# Generate x values
x = np.linspace(-10, 10, 100)

# Calculate corresponding y values
y = m * x + b

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=f'y = {m}x + {b}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Equation Plot')
plt.grid(True)
plt.legend()
plt.show()
