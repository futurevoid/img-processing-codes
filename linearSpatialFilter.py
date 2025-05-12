import cv2 
import numpy as np
import matplotlib.pyplot as plt

def linear_filter(img,kernel):
    filtered_img = cv2.filter2D(src=img,ddepth=-1,kernel=kernel)
    return filtered_img

img = cv2.imread("lion.jpg",cv2.IMREAD_GRAYSCALE)

kernel = np.ones((3,3),dtype=np.float32)/9

out = linear_filter(img, kernel)

plt.figure(figsize=(10,6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Filtered Image")
plt.imshow(out, cmap='gray')
plt.axis("off")


plt.tight_layout()
plt.show()