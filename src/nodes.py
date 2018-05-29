from matplotlib import pyplot as plt

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

def find_nodes(image):
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