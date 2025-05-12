from skimage import io, transform, color
import pylab as pl
import matplotlib.pyplot as plt
import cv2


img = cv2.imread("lion.jpg", 0)
rescaled = transform.rescale(img, 2, anti_aliasing=False)
resized = transform.resize(img,(900,900))
rotated = transform.rotate(img,45,(500,200))

img = cv2.imread("lion.jpg")
imghsv = color.rgb2hsv(img)
imgxyz = color.rgb2xyz(img)
imgypbpr = color.rgb2ypbpr(img)

cv2.imshow("rotated",rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.figure(figsize=(10,6))

plt.subplot(2,3,1)
plt.imshow(img)
plt.title("img")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(rescaled)
plt.title("rescaled")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(resized)
plt.title("resized")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(imghsv)
plt.title("imghsv")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(imgxyz)
plt.title("imgxyz")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(imgypbpr)
plt.title("imgypbpr")
plt.axis("off")


plt.show()

