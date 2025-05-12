import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lion.jpg",0)

fft = np.fft.fft2(img)
fft_shift = np.abs(np.fft.fftshift(fft))
magnitude_spectrum = np.log1p(fft_shift)

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("img")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(magnitude_spectrum,cmap="gray")
plt.title("magnitude_spectrum")
plt.axis("off")

plt.show()