import cv2
import numpy as np
import matplotlib.pyplot as plt

def glpf(shape, cutoff):
    rows, cols = shape
    center_rows, center_cols = rows // 2 , cols // 2
    
    rows_indices = np.arange(rows)
    cols_indices = np.arange(cols)
    cols_grid, rows_grid = np.meshgrid(cols_indices, rows_indices)
    
    dist_from_center = np.sqrt((cols_grid - center_cols)** 2 + (rows_grid - center_rows)** 2)
    
    filter_mask = np.exp(-(dist_from_center**2 / (2 * cutoff**2)))
    
    return filter_mask


def apply_glpf(img, cutoff):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    dft = np.fft.fft2(gray)
    dft_shifted = np.fft.fftshift(dft)
    
    gaussian = glpf(gray.shape,cutoff)
    
    filtered_dft = dft_shifted * gaussian
    
    inv_dft_shifted = np.fft.ifftshift(filtered_dft)
    image_reconstructed = np.abs(np.fft.ifft2(inv_dft_shifted))
    
    return gray, image_reconstructed 

img = cv2.imread("lion.jpg")

original_img, glpf_img = apply_glpf(img,cutoff=50)

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(original_img,cmap="gray")
plt.title("original_image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(glpf_img,cmap="gray")
plt.title("glpf_image")
plt.axis("off")

plt.show()
    