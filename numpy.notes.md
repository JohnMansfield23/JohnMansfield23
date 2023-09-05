# NumPy and Matplotlib Functions

This README provides explanations and examples for some commonly used NumPy and Matplotlib functions in Python.

## NumPy Functions

### `np.zeros`

The `np.zeros` function creates a NumPy array filled with zeros. You can specify the shape of the array as a tuple.

Example:

```python
import numpy as np

# Create a 2x3 array of zeros
zeros_array = np.zeros((2, 3))
print(zeros_array)
```
#### `np.ones`
```python
import numpy as np

# Create a 3x2 array of ones
ones_array = np.ones((3, 2))
print(ones_array)
```
#### `np.eye`
```python
import numpy as np

# Create a 4x4 identity matrix
identity_matrix = np.eye(4)
print(identity_matrix)
```
##### `linspace`
```python
import numpy as np

# Create an array of 10 evenly spaced values between 0 and 1
linspace_array = np.linspace(0, 1, 10)
print(linspace_array)
```
###### `Matoplotlib`
####### `plt.imshow`
```python
import matplotlib.pyplot as plt
import numpy as np

# Create a simple 2D array
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Display the array as an image
plt.imshow(data, cmap='viridis')
plt.colorbar()
plt.show()
```
