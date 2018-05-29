import cv2
from skimage import io 
from skimage.filters import threshold_otsu 
import os 
import numpy as np
from matplotlib import pyplot as plt  

def convert_binary_to_normal_im(binary):
    gray_img = ((1-binary)*255).astype('uint8')
    return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)  

def neighbours(x,y,image):
    img = image.copy() 
    row, column = img.shape
    x_left, x_right, y_down, y_up = neighbors = x-1, x+1, y-1, y+1 
    neighbours = -1 
    for i in range(x_left, x_right+1):
        for j in range(y_down, y_up+1):
            neighbours += image[j][i]

    adjacent = False
    if (all(i in range(0, column-1) for i in (x, x_left, x_right))):
        if (all(j in range(0, row-1) for j in (y, y_down, y_up))):
            adjacent = adjacentNeighbor(x,y,neighbors, image)
    return neighbours,adjacent

def adjacentNeighbor(x,y,neighbors, image):
    x_left, x_right, y_down, y_up = neighbors
    adjacentNeighbor = False
    if (image[y_up][x] == 1):
        if (image[y_up][x_left] == 1 or image[y_up][x_right] == 1):
            adjacentNeighbor = True
            return adjacentNeighbor
    if (image[y][x_right] == 1):
        if (image[y_down][x_right] == 1 or image[y_up][x_right] == 1):
            adjacentNeighbor = True
            return adjacentNeighbor
    if (image[y_down][x] == 1):
        if (image[y_down][x_left] == 1 or image[y_down][x_right] == 1):
            adjacentNeighbor = True
            return adjacentNeighbor
    if (image[y][x_left] == 1):
        if (image[y_down][x_left] == 1 or image[y_up][x_left] == 1):
            adjacentNeighbor = True
            return adjacentNeighbor

    return adjacentNeighbor

def findingNodes(image):
    "Return a dict of nodes"
    img = image.copy()
    nodes = []
    rows, columns = img.shape
    # print(rows, columns)
    for x in range(1, columns-1):
        for y in range(1, rows-1):
            if (img[y][x] == 1):
                n, ad = neighbours(x,y,img)
                if (not ad and n > 2):
                    nodes.append([x,y])
    return nodes

def show_img(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    cwd = os.getcwd()
    img_original = cv2.imread(cwd+"/m.png")
    if (len(np.shape(img_original)) == 3 and np.shape(img_original)[2] == 3):
        gray_image = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = img_original

    gray_image = np.array(~(gray_image>0)*1)
    print(gray_image)
    print(np.shape(gray_image))
    # print(h, w)
    n = findingNodes(gray_image)

    node_img = convert_binary_to_normal_im(gray_image)
    [cv2.circle(node_img, (col, row), 1, (0,255,0), -1) for col, row in n]
    cv2.imwrite("node_img.png", node_img)
    show_img(node_img)