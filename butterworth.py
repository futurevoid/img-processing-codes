import cv2 
import numpy as np
import matplotlib.pyplot as plt


def blpf(shape, cutoff, order):
    num_rows, num_cols = shape
    rows_indices = np.arange(num_rows)
    cols_indices = np.arange(num_cols)
    cols_grid, rows_grid = np.meshgrid(cols_indices, rows_indices)
    dist_from_center = np.sqrt(((cols_grid - num_cols // 2)**2 + (rows_grid - num_rows // 2)**2))
    filter_mask = 1 / (1 + (dist_from_center/cutoff)** (2 * order))
    return filter_mask

def apply_blpf(img, cutoff=30 , order=2):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    #fft
    dft = np.fft.fft2(img)
    dft_shifted = np.fft.fftshift(dft)
    
    #filter
    blpf_filter = blpf(img.shape,cutoff,order)
    filtered_dft = dft_shifted * blpf_filter
    
    #ifft
    inv_shifted = np.fft.ifftshift(filtered_dft)
    reconstructed_img = np.abs(np.fft.ifft2(inv_shifted))
    return reconstructed_img, blpf_filter

img1 = cv2.imread("lion.jpg")

filtered_img, butterworth = apply_blpf(img1,cutoff=50, order=2) 

plt.figure(figsize=(15,6))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Butterworth Filter")
plt.imshow(butterworth, cmap='gray')
plt.axis('off')


plt.subplot(1,3,3)
plt.title("Filtered Image")
plt.imshow(filtered_img, cmap='gray')
plt.axis("off")


plt.tight_layout()
plt.show()