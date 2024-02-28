import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt

# Function to perform KMeans color quantization
def kmeans_color_quantization(image_path, n_colors):
    # Open image
    img = Image.open(image_path)
    
    # Convert image to RGB (if it's not already)
    img = img.convert('RGB')
    
    # Convert image data to a numpy array
    img_data = np.array(img)
    
    # Reshape image data to a 2D array (each pixel is a row)
    pixels = img_data.reshape((-1, 3))
    
    # Perform KMeans clustering on the pixel data
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Get the centroids (the cluster centers) and labels (the cluster assignment for each pixel)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Map each pixel to the color of its cluster centroid
    quantized_pixels = centroids[labels]
    
    # Reshape quantized pixel data back to the image shape
    quantized_img_data = quantized_pixels.reshape(img_data.shape)
    
    # Convert back to an image
    quantized_img = Image.fromarray(np.uint8(quantized_img_data))
    
    return quantized_img


image_path = 'ai.webp'  
n_colors = 2  # Number of colors to reduce to

# Apply KMeans color quantization
quantized_image = kmeans_color_quantization(image_path, n_colors)

# Display original and quantized image
original_img = Image.open(image_path)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(quantized_image)
plt.title(f'Quantized Image ({n_colors} colors)')
plt.axis('off')

plt.show()

# Optionally, save the quantized image
quantized_image.save('quantized_image.jpg')