import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lion.jpg", 0)
fft = np.fft.fft2(img)
fft_shifted = np.fft.fftshift(fft)
mag_spectrum1 = np.abs(np.log1p(fft_shifted))

rows,cols = fft.shape
ideal_filter = np.zeros((rows,cols),dtype=np.float32)

center_freq = 12
bandwidth = 6

for row in range(rows):
    for col in range(cols):
        dist = np.sqrt((row - rows/2)**2 + (col - cols/2)**2 )
        if (center_freq-(bandwidth/2) <= dist <= center_freq +(bandwidth/2)):
            ideal_filter[row,col] = 0
        else:
            ideal_filter[row, col] = 1
            
filtered_fft_shifted = fft_shifted * ideal_filter
mag_spectrum2 = np.log1p(np.abs(filtered_fft_shifted))

inv_fft = np.fft.ifftshift(filtered_fft_shifted)
reconstructed_img = np.abs(np.fft.ifft2(inv_fft))


# Plot results
plt.figure(figsize=(10, 10))
plt.subplot(3, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Image')
plt.axis("off")
plt.subplot(3, 2, 2)
plt.imshow(mag_spectrum1, cmap='gray')
plt.title("Fourier of image")
plt.axis("off")
plt.subplot(3, 2, 3)
plt.imshow(reconstructed_img, cmap='gray')
plt.title("IBRF in image")
plt.axis("off")
plt.subplot(3, 2, 4)
plt.imshow(mag_spectrum2, cmap='gray')
plt.title("IBRF")
plt.axis("off")
plt.tight_layout()
plt.show()