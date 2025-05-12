import cv2 

def get_neighbors(img, neighbor_type='4'):
    img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found. check path")
    
    height, width = img.shape
    if neighbor_type == '4':
        neighbor_offsets = [(-1,0),(1,0),(0,-1),(0,1)]
    elif neighbor_type == '8':
        neighbor_offsets = [(-1,0),(1,0),(0,-1),(0,1),
                            (-1,-1),(-1,1),(1,-1),(1,1)]
    else:
        print("Invalid Type. Choose 4 or 8 neighbors")
        
    neighbors_dict = {}
    
    for row in range(height):
        for col in range(width):
            neighbors = []
            for offset_y, offset_x in neighbor_offsets:
                neighbor_row, neighbor_col = row + offset_y, col + offset_x
                if 0 <= neighbor_row < height and  0 <= neighbor_col < width:
                    neighbors.append((neighbor_row, neighbor_col))
            neighbors_dict[(row,col)] = neighbors
    return neighbors_dict

image = "lion.jpg"
neighbors = get_neighbors(image,'8')

print("neighbors of pixel (3,3):")
for neighbor in neighbors.get((3,3)):
    print(neighbor)