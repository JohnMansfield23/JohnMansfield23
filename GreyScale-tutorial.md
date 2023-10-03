```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the grayscale image
image_path = 'path_to_your_grayscale_image.png'
img = mpimg.imread(image_path)

# Display the image
plt.imshow(img, cmap='gray')
plt.axis('off')  # Turn off axis labels
plt.show()
