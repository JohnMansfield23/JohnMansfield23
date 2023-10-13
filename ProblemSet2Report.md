(Colab File Link)[https://colab.research.google.com/drive/1FTgzsr0Ioe-LRQtVdu5iYNicL2sCLNgw?usp=sharing]

# Image Processing Report

In this report, we demonstrate the steps of processing an image which includes its loading, resizing, grayscale conversion, and application of convolution filters to extract features. The image used for this demonstration is the logo of Florida Atlantic Owls.

## Step 1: Loading the Image

```python
URL = 'https://1000logos.net/wp-content/uploads/2019/12/Florida-Atlantic-Owls-Logo-1994.png'
response = requests.get(URL, stream=True)
img = Image.open(response.raw).convert("RGB")
```
### Step 2: Displaying the original image
```python
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')
plt.show()
```
#### Step 3: Resizing the Image
```python
resized_image = cv2.resize(img_array, (224, 224))
```
##### Step 4: Convert to greyscale
```python
grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
```
###### Step 5: Applying Random Filters
```python
filter_size = 3
filters = [np.random.randn(filter_size, filter_size) for _ in range(10)]
```
###### Step 6: Display filters and feature maps
```python
fig, axs = plt.subplots(10, 2, figsize=(10, 30))
for i, filt in enumerate(filters):
    feature_map = cv2.filter2D(grayscale_image, -1, filt)

    axs[i, 0].imshow(filt, cmap='gray')
    axs[i, 0].set_title(f"Filter {i + 1}")
    axs[i, 0].axis('off')

    axs[i, 1].imshow(feature_map, cmap='gray')
    axs[i, 1].set_title(f"Feature Map {i + 1}")
    axs[i, 1].axis('off')

plt.tight_layout()
plt.show()
```
