import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lion.jpg",cv2.IMREAD_GRAYSCALE)

eq_img = cv2.equalizeHist(img)

plt.figure(figsize=(10,10))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2,2,2)
plt.title("Equalized Image")
plt.imshow(eq_img, cmap='gray')
plt.axis("off")

plt.subplot(2,2,3)
plt.title("Original Image Histogram")
plt.hist(img.ravel(),256,[0,256])

plt.subplot(2,2,4)
plt.title("Equalized Image Histogram")
plt.hist(eq_img.ravel(),256,[0,256])

plt.tight_layout()
plt.show()