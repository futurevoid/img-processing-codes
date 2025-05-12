# Connected Components
import cv2
import numpy as np
# Loading the image
img = cv2.imread('lion.jpg')

# grayscale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#7x7 gaussian blur
blur = cv2.GaussianBlur(gray,(7,7),0)

#threshold
thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#component analysis
analysis = cv2.connectedComponentsWithStats(thresh,4,cv2.CV_32S)
(totalLabels, label_ids, vals, centroid) = analysis

out = np.zeros(gray.shape,dtype="uint8")

for lbl_id in range(1,totalLabels):
    area = vals[lbl_id, cv2.CC_STAT_AREA]
    if (area > 140) and (area < 400):
        ComponentMask = (label_ids == lbl_id).astype("uint8") * 255
        out = cv2.bitwise_or(out, ComponentMask)
        
cv2.imshow("img",img)
cv2.imshow("connected",out)
cv2.waitKey(0)
cv2.destroyAllWindows()