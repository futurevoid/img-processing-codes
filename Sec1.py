import cv2
import numpy as np

img = cv2.imread("lion.jpg")
gray = cv2.imread("liongray.jpg")
resized = cv2.resize(img, (410,340))

(h,w) = img.shape[:2]
center = (w//2, h//2)
scale = 1.0
Mat2D = cv2.getRotationMatrix2D(center, 45, scale)
rotated45 = cv2.warpAffine(img, Mat2D, (h,w))

rotated90ct = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
rotated90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
rotated180 = cv2.rotate(img, cv2.ROTATE_180)

circle_img = cv2.circle(resized, (220,150), 100, (250,234,120),6)
line_img = cv2.line(resized, (220,150), (100,100), (250,129,120),8)
rect_img = cv2.rectangle(resized, (225,235), (300,200), (200,109,130),10)

pixel = img[100,102]
img[100,102] = [0,0,255]

cv2.putText(resized, 'LION',[20,25],cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,1),2,cv2.LINE_AA)

B,G,R = cv2.split(resized)

WeightedSum = cv2.addWeighted(img,0.01,gray,0.99,0.9)

sub = cv2.subtract(img,gray)

bitand = cv2.bitwise_and(img,gray)
bitor = cv2.bitwise_or(img,gray)
bitxor = cv2.bitwise_xor(img,gray)
bitnot = cv2.bitwise_not(img, mask=None)
bitnot1 = cv2.bitwise_not(gray, mask=None)

kernel = np.ones((20,20),np.uint8)
eroded = cv2.erode(img,kernel)

gaussian = cv2.GaussianBlur(resized, (9,9),0)

median = cv2.medianBlur(resized, 5)

bilateral = cv2.bilateralFilter(resized, 9, 75, 75)

image_with_border = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[200, 100, 50])
# cv2.imshow('image',img)
# cv2.imshow('gray',gray)
# cv2.imshow('resized',resized)
# cv2.imshow('rotated45',rotated45)
# cv2.imshow('rotated90ct',rotated90ct)
# cv2.imshow('rotated90',rotated90)
# cv2.imshow('rotated180',rotated180)
# cv2.imshow("circle_img",circle_img)
# cv2.imshow("line_img",line_img)
# cv2.imshow("rect_img",rect_img)
# cv2.imshow("BLUE",B)
# cv2.imshow("GREEN",G)
# cv2.imshow("RED",R)
# cv2.imshow("WeightedSum",WeightedSum)
# cv2.imshow("sub",sub)
# cv2.imshow("bitand",bitand)
# cv2.imshow("bitor",bitor)
# cv2.imshow("bitxor",bitxor)
# cv2.imshow("bitnot",bitnot)
# cv2.imshow("bitnot1",bitnot1)
# cv2.imshow("eroded",eroded)
cv2.imshow("gaussian",gaussian)
cv2.imshow("median",median)
cv2.imshow("bilateral",bilateral)
cv2.imshow("image_with_border",image_with_border)

#cv2.imwrite("liongray.jpg",gray)
cv2.waitKey(0)
cv2.destroyAllWindows()