import matplotlib.pyplot as plt
from skimage import util, img_as_uint, io, morphology
import numpy as np
import cv2
from matplotlib import pyplot
import skimage.color as color
from skimage.filters import threshold_otsu
import copy
import os
import sknw
import networkx as nx
from scipy.spatial import Voronoi, voronoi_plot_2d
from strokeWidth import stroke_width_transform  
from nodes import findingNodes

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

def circle_image(img, coordinates, name = "circle_image", num=0):
    img_out = convert_binary_to_normal_im(img)
    [cv2.circle(img_out, (col, row), 1, (0,255,0), -1) for row, col in coordinates]
    cv2.imwrite(cwd+"/debug/{}_{}.png".format(name, num), img_out)

def line_image(img, coordinates, name = "debug", num=0):
    img_out = convert_binary_to_normal_im(image)
    height, width = np.shape(img)
    [cv2.line(img_out, (col,0), (col, height-1), (255,0,0), 3, -1) for col,_ in coordinates]
    cv2.imwrite(cwd + "/debug/{}_{}.png".format(name, num), img_out)

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

def get_node_coordinate(graph, node):
    positions, nodes = graph.node, graph.nodes()
    coordinates = positions[node]['pts'][:]
    return coordinates

# def get_segments(image, original_image, averageStroke=0):
#     gray_img = ((1-image)*255).astype('uint8')
#     x, y, width, height = cv2.boundingRect(255-gray_img)
#     color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
#     cv2.rectangle(color_image, (x,y), (x+width, y+height), thickness=3, color=(255,0,0))
#     cv2.imwrite("abcd.png", color_image)

#     # height, width = np.shape(image)
#     print(width, height)
#     maxWid = 1*height
#     minWid = 0.7*height
#     projection = np.array([sum(image[:,i]) for i in range(width)]) - averageStroke
#     projection[projection < 0] = 0
#     zeroPoints = [i for i in range(len(projection)) if projection[i] == 0]
#     seperatingPoints  = getseperatingPoints (zeroPoints)
#     seperatingPoints  = [point for point in seperatingPoints  if point > 0]

#     output = convert_binary_to_normal_im(image)
#     [cv2.line(output, (col,0), (col, height-1), (255,0,0), 3, -1) for col in seperatingPoints ]
#     cv2.imwrite(cwd + "/debug/sperablePoints.png", output)

#     mediumsized SegmentPoints = [[seperatingPoints [i-1], seperatingPoints [i]] 
#     for i in range(1, len(seperatingPoints )) if (minWid < (seperatingPoints [i] - seperatingPoints [i-1]) < maxWid)]
#     oversizedSegmentPoints = [[seperatingPoints [i-1], seperatingPoints [i]] 
#     for i in range(1, len(seperatingPoints )) if ((seperatingPoints [i] - seperatingPoints [i-1]) > maxWid)]
    
#     mediumsized Segments = [original_image[0:width, i:j] for i,j in mediumsized SegmentPoints]
#     oversizedSegments = [original_image[0:width, i:j] for i,j in oversizedSegmentPoints]

#     return mediumsized Segments, oversizedSegments

def get_segments(image, original_image, averageStroke=0):
    width, height = get_img_shape(image)
    # print(width, height)
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
    # cv2.imwrite(cwd + "/debug/line_image.png", lineImage)
    # show_img(lineImage)

    cutPointSet = [[seperatingPoints [i-1], seperatingPoints [i]] for i in range(1, len(seperatingPoints))]
    cutImages = [original_image[0:width, i:j] for i,j in cutPointSet]
    # [show_img(seg, "Segments") for seg in cutImages]
    mediumsizedSegments = [seg for seg in cutImages if (minWid < get_img_shape(seg)[0] < maxWid)]
    # [show_img(mediumsized , "mediumsized  segments") for mediumsized  in mediumsizedSegments]
    oversizedSegments = [seg for seg in cutImages if (get_img_shape(seg)[0] > maxWid)]
    # [show_img(oversized, "oversized segments") for oversized in oversizedSegments]
    # undersizedSegments = [seg for seg in cutImages if (get_img_shape(seg)[0] < minWid)]
    # [show_img(undersized, "undersized segments") for undersized in undersizedSegments]
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
    forkPoints = findingNodes(img)
    try: 
        path = nx.astar_path(graph, src, des)[1:-1]
        coordinates = np.array([coor for p in path for coor in get_node_coordinate(graph, p)]).tolist()
        forkPoints = np.array(findingNodes(segment_skeleton)).tolist()
        forkPoints = [p[::-1] for p in forkPoints]
        voronoi_points = [coors for coors in coordinates if coors in forkPoints]
        result.append(voronoi_points)
        circle_image(img, coordinates, "bridge", numOfBridges)
        # print(voronoi_points)

        for row, col in coordinates:
            img[row][col] = 0
        debug = convert_binary_to_normal_im(img)
        cv2.imwrite(cwd+"/debug/debug.png", debug)
    except:
        return result
    numOfBridges += 1
    # return find_bridges_between_VLs(img, [src,des], centerOfVLs, result, numOfBridges)
    
if __name__ == '__main__':
    cwd = os.getcwd()
    if not os.path.exists(os.path.join(os.getcwd(), 'debug')):
        os.mkdir(os.path.join(os.getcwd(), 'debug'))
    os.system('rm -r ' + cwd + "/debug/*")
    os.mkdir(os.path.join(os.getcwd(), 'debug/mediumsizedSegments'))
    os.mkdir(os.path.join(os.getcwd(), 'debug/overSegments'))
    os.mkdir(os.path.join(os.getcwd(), 'debug/touchingCases'))
    path = cwd + "/tight/tight7.png"
    Img_Original = io.imread(path)
    if (len(np.shape(Img_Original)) == 3 and np.shape(Img_Original)[2] == 3):
        gray_image = cv2.cvtColor(Img_Original, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = Img_Original
    x, y, width, height = component = cv2.boundingRect(255-gray_image)
    color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(color_image, (x,y), (x+width, y+height), thickness=3, color=(255,0,0))
    cv2.imwrite(cwd+"/debug/bouding_box_image.png", color_image)
    # show_img(gray_image)
    Otsu_Threshold = threshold_otsu(gray_image)
    BW_Original = (gray_image < Otsu_Threshold) > 0
    # BW_Original = util.invert(BW_Original)
    io.imsave(cwd+"/debug/binary.png", img_as_uint(BW_Original))

    """ 
    ...................... Coarse Segmentation ................

    """
    averageStroke = int(stroke_width_transform.scrub(path))
    # print(averageStroke)
    kernel = np.ones((8,1), np.uint8)
    closingImg = cv2.morphologyEx(img_as_uint(BW_Original), cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(cwd+"/debug/closing_image.png",closingImg)
    # show_img(closingImg)
    closingImg = closingImg > 0

    overSegments, mediumSegments = get_segments(closingImg, BW_Original)
    [cv2.imwrite(cwd+"/debug/overSegments/oversized_segments_{}.png".format(i), convert_binary_to_normal_im(overSegments[i])) for i in range(len(overSegments))]
    [cv2.imwrite(cwd+"/debug/mediumsizedSegments/mediumsized_segments_{}.png".format(i), convert_binary_to_normal_im(mediumSegments[i])) for i in range(len(mediumSegments))]

    voronoi_cases = [] 
    for index in range(len(overSegments)):
        # show_img(overSegments[index], "Over segments")
        touchingCases, untouchingCases = get_segments(overSegments[index], overSegments[index], averageStroke)
        voronoi_cases.append(touchingCases)
        [cv2.imwrite(cwd+"/debug/mediumsizedSegments/untouchingCases_{}_{}.png".format(index,i), convert_binary_to_normal_im(untouchingCases[i])) for i in range(len(untouchingCases))]
        [cv2.imwrite(cwd+"/debug/touchingCases/touchingCases_{}_{}.png".format(index,i), convert_binary_to_normal_im(touchingCases[i])) for i in range(len(touchingCases))]
        # [show_img(case, "untouchingCases") for case in untouchingCases]
        # [show_img(case, "touchingCases") for case in touchingCases]

    [show_img(img) for cases in voronoi_cases for img in cases]

    """ 
    ................  fine segmentation ...................

    """ 
    # Step 1: Thin each oversized segment
    # size = np.shape(overSegments[1])
    # AveWid = size[0]*0.8
    # segment_skeleton = morphology.skeletonize(overSegments[1]).astype(np.uint16)

    # # Step 2: Draw vertical lines (VLs)
    # h, w = shape = np.shape(segment_skeleton)
    # projection = [sum(segment_skeleton[:,i]) for i in range(w)]
    # # plt.plot(projection)
    # # plt.show()
    # nMaximumPeaks = np.sort(projection)[w-int(np.ceil(w/AveWid)):w]
    # # nMaximumPeaks = np.sort(projection)[w-10:w]
    # # print(nMaximumPeaks)
    # # maximumPeakPos = list(set([projection.index(i) for i in nMaximumPeaks]))
    # maximumPeakPos = []
    # for i in range(len(projection)):
    #     if projection[i] in nMaximumPeaks:
    #         maximumPeakPos.append(i)
    # # print(maximumPeakPos)

    # for point in maximumPeakPos:
    #     segment_skeleton[:, point] = 1
    #     segment_skeleton[int(h/2), point+1] = 1
    #     segment_skeleton[int(h/2), point-1] = 1

    # # cv2.imwrite(cwd+"/debug/VL_image.png", convert_binary_to_normal_im(segment_skeleton))
    # # show_img(segment_skeleton)

    # # Step 3: Seek and seperate the shortest bridges between every VL pair
    # graph = sknw.build_sknw(segment_skeleton)
    # node, nodes = graph.node, graph.nodes()
    # coordinates = [get_node_coordinate(graph,n) for n in nodes]
    # centerOfVLs = [[int(h/2),p] for p in maximumPeakPos]

    # nodesOfVLs = [] 
    # for point in centerOfVLs:
    #     for index in range(len(coordinates)):
    #         if (point in coordinates[index].tolist()):
    #             nodesOfVLs.append(index)

    # img = segment_skeleton.copy() 
    # print(nodesOfVLs)
    # for i in range(1,len(nodesOfVLs)):
    #     vorPoints = find_bridges_between_VLs(img, [nodesOfVLs[i-1], nodesOfVLs[i]], centerOfVLs)
    #     # print(nodesOfVLs[i-1], nodesOfVLs[i])
    #     # print(vorPoints)
 
    #     # Step 4: Seperated by Voronoi diagrams
    #     pointSet = vorPoints[0]
    #     # print(pointSet)
    #     try:
    #         vor = Voronoi(pointSet)
    #         voronoi_plot_2d(vor)
    #         plt.show()
    #         voronoi_points = (vor.vertices).astype(int)
    #         # circle_image(segment_skeleton, voronoi_points, "voronoi_img")
    #     except:
    #         print("Points are less than 3!")

    # vorPoints = find_bridges_between_VLs(img, [25,22], centerOfVLs)
    # print(vorPoints)

    # # Step 4: Seperated by Voronoi diagrams
    # pointSet = vorPoints[0]
    # print(pointSet)
    # vor = Voronoi(pointSet)
    # voronoi_points = (vor.vertices).astype(int)
    # # print("Points:{}".format(vor.points))
    # # print("Ridge points: {}".format(vor.ridge_points))
    # # print("Vertices:{}".format(vor.vertices))
    # # print("Ridge vertices: {}".format(vor.ridge_vertices))
    # # print("Regions: {}".format(vor.ridge_vertices))
    # # print("Point region: {}".format(vor.point_region))
    # voronoi_plot_2d(vor)
    # plt.show()
    # circle_image(segment_skeleton, voronoi_points, "voronoi_img")

    # Step 5: Reflect the segmentation boundaries to the original image
