import cv2
import numpy as np

def get_adj_mat(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found. check path")
    
    _,binary_img = cv2.threshold(img,128,1,cv2.THRESH_BINARY)
    
    rows,cols = img.shape
    print(f"rows:{rows} , cols:{cols}")
    
    adj_mat = np.zeros((rows*cols,rows*cols),dtype=int)
    
    def to_idx(row,col):
        return row*cols + col
    
    for row in range(rows):
        for col in range(cols):
            if binary_img[row,col] == 1:
                curr_idx = to_idx(row,col)
                
                if col + 1 < cols and binary_img[row,col] == 1:
                    right_idx = to_idx(row,col)
                    adj_mat[curr_idx,right_idx] = 1
                    adj_mat[right_idx,curr_idx] = 1
                elif row + 1 < rows and binary_img[row,col] == 1:
                    bottom_idx = to_idx(row,col)
                    adj_mat[curr_idx,bottom_idx] = 1
                    adj_mat[bottom_idx,curr_idx] = 1
    return adj_mat

img = "lion.jpg"

adj_mat = get_adj_mat(img)

print(adj_mat)