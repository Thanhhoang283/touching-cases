import matplotlib.pyplot as plt
from skimage import util, img_as_uint, io, morphology
import numpy as np
import cv2
from matplotlib import pyplot
import skimage.color as color
from skimage.filters import threshold_otsu
import os
import sknw
import networkx as nx
import voronoi
from strokeWidth import stroke_width_transform
from nodes import find_nodes
import argparse

def convert_binary_to_normal_im(binary):
    gray_img = ((1-binary)*255).astype('uint8')
    return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

def show_img(img, winname="DEBUG"):
    pyplot.figure()
    pyplot.imshow(img)
    pyplot.title(winname)
    pyplot.show()

def get_img_shape(image):
    "return size of binary image's bouding box"
    gray_img = ((1-image)*255).astype('uint8')
    x, y, width, height = cv2.boundingRect(255-gray_img)
    # color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    # cv2.rectangle(color_image, (x,y), (x+width, y+height), thickness=3, color=(255,0,0))
    # show_img(color_image)
    return width, height

def draw_point(img, coordinates, name = "draw point", num=0):
    """ Parameters: 
    img: binary image
    coordinates: 2-dimens numpy array 
    """
    img_out = convert_binary_to_normal_im(img)
    [cv2.circle(img_out, (col, row), 1, (0,255,0), -1) for row, col in coordinates]
    cv2.imwrite("{}_{}.png".format(name, num), img_out)
    show_img(img_out)

# def draw_line(img, coordinates, name = "draw line", num=0):
#     img_out = convert_binary_to_normal_im(img)
#     height, width = np.shape(img)
#     [cv2.line(img_out, (col,0), (col, height-1), (255,0,0), 2, -1) for _,col in coordinates]
#     cv2.imwrite("{}_{}.png".format(name, num), img_out)
#     show_img(img_out)

def draw_line(img, coordinates, name = "draw line", num=0):
    img_out = convert_binary_to_normal_im(img)
    height, width = np.shape(img)
    [cv2.line(img_out,(coordinates[i-1][1], coordinates[i-1][0]),  
        (coordinates[i][1], coordinates[i][0]), 
        (255,0,0), 2, -1) for i in range(1,len(coordinates))]
    cv2.imwrite("{}_{}.png".format(name, num), img_out)
    show_img(img_out)

def getseperatingPoints (points,up=0,down=1):
    result = []
    for i in range(down, len(points)):
        if ((points[i]-1) != points[i-1]):
            up = i-1
            result.append(np.mean(points[down:up], dtype=int))
            down = i
        elif (i == (len(points)-1)):
            result.append(np.mean(points[down:i], dtype=int))
    return result

def preprocess_image(original_image):
    if (len(np.shape(original_image)) == 3 and np.shape(original_image)[2] == 3):
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = original_image
    # x, y, width, height = component = cv2.boundingRect(255-gray_image)
    # color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    # cv2.rectangle(color_image, (x,y), (x+width, y+height), thickness=3, color=(255,0,0))
    # cv2.imwrite(os.path.join(directory, filename[:6]+"/debug/bouding_box_image.png"), color_image)
    # # show_img(gray_image)
    otsu_threshold = threshold_otsu(gray_image)
    binary_image = (gray_image < otsu_threshold) > 0
    return binary_image

def get_node_coordinate(graph, node):
    positions, nodes = graph.node, graph.nodes()
    coordinates = positions[node]['pts'][:]
    return coordinates

def check_touching_image(image, original_image, averageStroke=0):
    width, height = get_img_shape(image)
    projection = np.array([sum(image[:,i]) for i in range(width)]) - averageStroke
    projection[projection < 0] = 0
    zeroPoints = [i for i in range(len(projection)) if projection[i] == 0]
    if zeroPoints:
        return get_segments(image, original_image, averageStroke)
    else:
        return original_image, []

def get_segments(image, original_image, averageStroke=0):
    width, height = get_img_shape(image)
    maxWid = 1.0*height
    minWid = 0.3*height
    projection = np.array([sum(image[:,i]) for i in range(width)]) - averageStroke
    projection[projection < 0] = 0
    zeroPoints = [i for i in range(len(projection)) if projection[i] == 0]

    seperatingPoints  = getseperatingPoints(zeroPoints)
    seperatingPoints = combine_small_segments([point for point in seperatingPoints  if point > 0], maxWid, minWid)
    seperatingPoints.append(image.shape[1]-1)
    seperatingPoints.append(0)
    seperatingPoints = np.sort(seperatingPoints)

    lineImage = convert_binary_to_normal_im(image)
    [cv2.line(lineImage, (col,0), (col, image.shape[0]-1), (255,0,0), 3, -1) for col in seperatingPoints]
    # show_img(lineImage)

    cutPointSet = [[seperatingPoints [i-1], seperatingPoints [i]] for i in range(1, len(seperatingPoints))]
    cutImages = [original_image[0:width, i:j] for i,j in cutPointSet]
    # [show_img(seg, "Segments") for seg in cutImages]
    mediumsizedSegments = [seg for seg in cutImages if (minWid < get_img_shape(seg)[0] < maxWid)]
    # [show_img(mediumsized , "mediumsized  segments") for mediumsized  in mediumsizedSegments]
    oversizedSegments = [seg for seg in cutImages if (get_img_shape(seg)[0] > maxWid)]
    # [show_img(oversized, "oversized segments") for oversized in oversizedSegments]
    return oversizedSegments, mediumsizedSegments

def combine_small_segments(seperatingPoints, maxWidth, minWidth):
    begin = seperatingPoints[0]
    newSeperatingPoints = []
    newSeperatingPoints.append(begin)
    for point in range(1, len(seperatingPoints)):
        distance = abs(begin - seperatingPoints[point])
        if ((distance > maxWidth) or (minWidth < distance < maxWidth)):
            begin = seperatingPoints[point]
            newSeperatingPoints.append(begin)
        else:
            continue
    return newSeperatingPoints

def find_bridges_between_VLs(img, points, centerOfVLs, result=[], numOfBridges=0):
    src, des = points
    graph = sknw.build_sknw(img)
    # forkPoints = find_nodes(img)
    show_img(img)
    try:
        print("Finding the path...")
        path = nx.astar_path(graph, src, des)[1:-1]
        coordinates = np.array([coor for p in path for coor in get_node_coordinate(graph, p)]).tolist()
        forkPoints = np.array(find_nodes(img)).tolist()
        forkPoints = [p[::-1] for p in forkPoints]
        voronoi_points = [coors for coors in coordinates if coors in forkPoints]
        print(voronoi_points)
        result.append(voronoi_points)
        for row, col in coordinates:
            img[row][col] = 0
        debug = convert_binary_to_normal_im(img)
        cv2.imwrite("removed_nodes_{}.png".format(numOfBridges), debug)
    except:
        print("Result: ", result)
        return result
    numOfBridges += 1
    return find_bridges_between_VLs(img, [src,des], centerOfVLs, result, numOfBridges)

def coarse_segmentation(file,filename):
    directory = os.path.split(os.path.dirname(file))[0] + "/"
    path = file+filename
    print(filename)

    if not os.path.exists(os.path.join(directory, filename[:6])):
        os.mkdir(os.path.join(directory, filename[:6]))
    else:
        os.system('rm -r ' + os.path.join(directory, filename[:6]+"/*"))
    os.mkdir(os.path.join(directory, filename[:6]+'/debug'))
    os.mkdir(os.path.join(directory, filename[:6]+'/debug/mediumsizedSegments'))
    os.mkdir(os.path.join(directory, filename[:6]+'/debug/overSegments'))
    os.mkdir(os.path.join(directory, filename[:6]+'/debug/touchingCases'))

    image = preprocess_image(path)
    cv2.imwrite(os.path.join(directory, filename[:6]+"/debug/binary_image.png"), convert_binary_to_normal_im(image))
    averageStroke = int(stroke_width_transform.scrub(path))
    kernel = np.ones((8,1), np.uint8)
    closingImg = cv2.morphologyEx(img_as_uint(image), cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(directory, filename[:6]+"/debug/closing_image.png"),closingImg)
    # show_img(closingImg)
    closingImg = closingImg > 0

    overSegments, mediumSegments = check_touching_image(closingImg, image)
    [cv2.imwrite(os.path.join(directory, filename[:6]+"/debug/overSegments/oversized_segments_{}.png".format(i)), 
        convert_binary_to_normal_im(overSegments[i])) for i in range(len(overSegments))]
    [cv2.imwrite(os.path.join(directory, filename[:6]+"/debug/mediumsizedSegments/mediumsized_segments_{}.png".format(i)), 
        convert_binary_to_normal_im(mediumSegments[i])) for i in range(len(mediumSegments))]

    voronoi_cases = []
    for index in range(len(overSegments)):
        # show_img(overSegments[index], "Over segments")
        touchingCases, untouchingCases = check_touching_image(overSegments[index], overSegments[index], averageStroke)
        voronoi_cases.append(touchingCases)
        [cv2.imwrite(os.path.join(directory, filename[:6]+"/debug/mediumsizedSegments/untouchingCases_{}_{}.png".format(index,i)), 
            convert_binary_to_normal_im(untouchingCases[i])) for i in range(len(untouchingCases))]
        [cv2.imwrite(os.path.join(directory, filename[:6]+"/debug/touchingCases/touchingCases_{}_{}.png".format(index,i)), 
            convert_binary_to_normal_im(touchingCases[i])) for i in range(len(touchingCases))]
        # [show_img(case, "untouchingCases") for case in untouchingCases]
        # [show_img(case, "touchingCases") for case in touchingCases]
    # [show_img(img) for cases in voronoi_cases for img in cases]
    return voronoi_cases

def fine_segmentation(image):
    # Step 1: Thin each oversized segment
    size = get_img_shape(image)
    AveWid = size[1]*0.8
    segment_skeleton = morphology.skeletonize(image).astype(np.uint16)
    # show_img(segment_skeleton, "Skeleton")

    # Step 2: Draw vertical lines (VLs)
    h, w = shape = np.shape(segment_skeleton)
    # kernel = np.ones((8,1), np.uint8)
    # closingImg = cv2.morphologyEx(img_as_uint(segment_skeleton), cv2.MORPH_CLOSE, kernel)
    projection = [sum(segment_skeleton[:,i]) for i in range(w)]
    # plt.plot(projection)
    # plt.show()
    nMaximumPeaks = np.sort(projection)[w-int(np.ceil(w/AveWid)):w]
    maximumPeakPos = list(set([projection.index(i) for i in nMaximumPeaks]))

    for point in maximumPeakPos:
        segment_skeleton[:, point] = 1
        segment_skeleton[int(h/2), point+1] = 1
        segment_skeleton[int(h/2), point-1] = 1

    # show_img(segment_skeleton,"Skeleton with VLs")
    # Step 3: Seek and seperate the shortest bridges between every VL pair
    graph = sknw.build_sknw(segment_skeleton)
    node, nodes = graph.node, graph.nodes()
    coordinates = [get_node_coordinate(graph,n) for n in nodes]
    centerOfVLs = [[int(h/2),p] for p in maximumPeakPos]

    nodesOfVLs = []
    for point in centerOfVLs:
        for index in range(len(coordinates)):
            if (point in coordinates[index].tolist()):
                nodesOfVLs.append(index)
    print("Center of VLs: ", np.sort(nodesOfVLs))
    # draw_point(segment_skeleton, [get_node_coordinate(graph,n) for n in [27]])
    vorPoints = find_bridges_between_VLs(segment_skeleton.copy(), [27,28], centerOfVLs)
    # if (len(vorPoints[0])<3):
        # vorPoints[0].append([0,0])
    # print(vorPoints)
    # Step 4: Seperated by Voronoi diagrams
    # seperating_points = [voronoi.voronoi_edges(setPoints) for setPoints in vorPoints]
    # return seperating_points
    return 0

def add_padding_to_image(image, bordersize=10):
    image = cv2.imread(path)
    row, col = image.shape[:2]
    bottom = image[row-2:row, 0:col]
    mean= cv2.mean(bottom)[0]
    padding_image = cv2.copyMakeBorder(image, top=bordersize, 
        bottom=bordersize, left=bordersize, right = bordersize, 
        borderType = cv2.BORDER_CONSTANT, value=[mean,mean,mean])
    # show_img(image, "Image")
    # show_img(padding_image, "New image")
    # show_img(bottom, "bottom")
    return padding_image

def change_coordinates(coordinates, oldCenter, newCenter=[0,0]):
    lamdaX = abs(oldCenter[0] - newCenter[0])
    lamdaY = abs(newCenter[1] - oldCenter[1])
    new_coordinates = []
    for coordinate in coordinates:
        x = coordinate[0] - lamdaX
        y = coordinate[1] + lamdaY
        new_coordinates.append([x,y])
    print(new_coordinates)
    return new_coordinates

if __name__ == '__main__':
    DEFAULT_DATA_DIR = "/home/thanh/Desktop/segmentation/images/"
    parser = argparse.ArgumentParser(description="Strike through detection")
    parser.add_argument('--data', help="data directory", type=str, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    path = args.data
    # [coarse_segmentation(path,filename) for filename in os.listdir(path)]
    path = "/home/thanh/Desktop/cinnamon/segmentation/samples/sample1.png"
    image = cv2.imread(path)
    processed_image = preprocess_image(image.copy())
    lines = fine_segmentation(processed_image)
    print("Lines: ",lines)
    # for line in lines[0]:
    #     print("Line: ", line)
    #     draw_point(processed_image.copy(), line)
    #     draw_line(processed_image.copy(), line)





