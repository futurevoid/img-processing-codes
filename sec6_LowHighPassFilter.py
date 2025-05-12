import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lion.jpg", 0)
fft = np.fft.fft2(img)
fft_shifted = np.fft.fftshift(fft)
mag_spectrum1 = np.abs(np.log1p(fft_shifted))

cutoff_radius = 15

rows, cols = img.shape
center_row, center_col = rows // 2, cols // 2
LPF = np.zeros((rows, cols), np.uint8)
for row in range(rows):
    for col in range(cols):
        dist = np.sqrt((row - center_row)**2 + (col - center_col)**2)
        if dist <= cutoff_radius:
            LPF[row,col] = 1

filtered_fft_shifted_l = fft_shifted * LPF
mag_spectrum2 = np.log1p(np.abs(filtered_fft_shifted_l))

inv_fft_shifted_l = np.fft.ifftshift(filtered_fft_shifted_l)
img_reconstructed_lp = np.abs(np.fft.ifft2(inv_fft_shifted_l))

HPF = 1- LPF
filtered_fft_shifted_h = fft_shifted * HPF
mag_spectrum3 = np.log1p(np.abs(filtered_fft_shifted_h))

inv_fft_shifted_h = np.fft.ifftshift(filtered_fft_shifted_h)
img_reconstructed_hp = np.abs(np.fft.ifft2(inv_fft_shifted_h))

# Create the first figure with 3 rows and 2 columns
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
plt.imshow(img_reconstructed_lp, cmap='gray')
plt.title("Image with low pass filter when D0=15")
plt.axis("off")

plt.subplot(3, 2, 4)
plt.imshow(mag_spectrum2, cmap='gray')
plt.title("Fourier transform image with low pass filter when D0=15")
plt.axis('off')

plt.subplot(3, 2, 5)
plt.imshow(img_reconstructed_hp, cmap='gray')
plt.title("Image with high pass filter when D0=15")
plt.axis("off")

plt.subplot(3, 2, 6)
plt.imshow(mag_spectrum3, cmap='gray')
plt.title("Fourier transform image with high pass filter when D0=15")
plt.axis('off')
plt.tight_layout()
plt.show()

# Create the second figure with filter transfer functions
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(LPF, cmap='gray')
plt.title("Low pass transfer function")


plt.subplot(1, 2, 2)
plt.imshow(HPF, cmap='gray')
plt.title("High pass transfer function")

plt.tight_layout()
plt.show()