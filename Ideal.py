import cv2
import numpy as np
import matplotlib.pyplot as plt

def ilpf(img, cutoff_radius):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    num_rows, num_cols = gray.shape
    center_row, center_col = num_rows // 2 , num_cols // 2
    
    dft = np.fft.fft2(gray)
    dft_shifted = np.fft.fftshift(dft)
    
    filter_mask = np.zeros((num_rows, num_cols),np.uint8)
    for row in range(num_rows):
        for col in range(num_cols):
            dist = np.sqrt((row - center_row)**2 + (col - center_col)**2) 
            if dist <= cutoff_radius:
                filter_mask[row,col] = 1
    
    filtered_dft = dft_shifted * filter_mask
    inv_shifted = np.fft.ifftshift(filtered_dft)
    reconstructedimg = np.abs(np.fft.ifft2(inv_shifted))
    return gray, reconstructedimg, filter_mask

img = cv2.imread("lion.jpg")

original_img, filtered_img, filter_mask = ilpf(img,50)

plt.figure(figsize=(10,6))

plt.subplot(1,3,1)
plt.imshow(original_img, cmap="gray")
plt.title("original_img")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(filtered_img, cmap="gray")
plt.title("filtered_img")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(filter_mask,cmap="gray")
plt.title("Ideal Low Pass Transfer Function")

plt.show()