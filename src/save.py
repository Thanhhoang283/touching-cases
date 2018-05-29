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
from strokeWidth import SWTScrubber  
from nodes import findingNodes

def convert_binary_to_normal_im(binary):
    gray_img = ((1-binary)*255).astype('uint8')
    return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

def show_img(img, winname="TEST"):
    pyplot.figure()
    pyplot.imshow(img)
    pyplot.title(winname)
    pyplot.show()

def getSeperablePoints(points,up=0,down=1):
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

def get_segments(image, averageStroke=0):
    height, width = np.shape(image)
    maxWid = 1.2*height
    minWid = 0.5*height
    projection = np.array([sum(image[:,i]) for i in range(width)]) - averageStroke
    projection[projection < 0] = 0
    zeroPoints = [i for i in range(len(projection)) if projection[i] == 0]
    seperablePoints = getSeperablePoints(zeroPoints)
    seperablePoints = [point for point in seperablePoints if point > 0]

    output = convert_binary_to_normal_im(image)
    [cv2.line(output, (col,0), (col, height-1), (255,0,0), 3, -1) for col in seperablePoints]
    # show_img(output)
    cv2.imwrite(cwd + "/debug/sperablePoints.png", output)

    fineSegmentPoints = [[seperablePoints[i-1], seperablePoints[i]] 
    for i in range(1, len(seperablePoints)) if (minWid < (seperablePoints[i] - seperablePoints[i-1]) < maxWid)]
    oversizedSegmentPoints = [[seperablePoints[i-1], seperablePoints[i]] 
    for i in range(1, len(seperablePoints)) if ((seperablePoints[i] - seperablePoints[i-1]) > maxWid)]
    
    fineSegments = [image[0:width, i:j] for i,j in fineSegmentPoints]
    oversizedSegments = [image[0:width, i:j] for i,j in oversizedSegmentPoints]
    return fineSegments, oversizedSegments

def find_bridges_between_VLs(img, graph, points, result=[], num=0):
    src, des = 25, 22
    try: 
        path = nx.astar_path(graph, src, des)
        result.append(path)
        coordinates = [coor for p in path for coor in get_node_coordinate(graph, p)]
        img_out = convert_binary_to_normal_im(img)
        [cv2.circle(img_out, (row, col), 1, (0,255,0), -1) for row, col in coordinates]
        cv2.imwrite(cwd+"/debug/path_{}.png".format(num), img_out)
    except:
        print("Result:",result)
        return result

    node_img = convert_binary_to_normal_im(segment_skeleton)
    [cv2.circle(node_img, (col, row), 1, (0,255,0), -1) for row, col in coordinates]
    cv2.imwrite(cwd+"/debug/node_path.png", node_img)
    num += 1
    return find_bridges_between_VLs(graph, points, coordinates, result, num)
    
if __name__ == '__main__':
    cwd = os.getcwd()
    if not os.path.exists(os.path.join(os.getcwd(), 'debug')):
        os.mkdir(os.path.join(os.getcwd(), 'debug'))
    os.system('rm -f ' + cwd + "/debug/*")
    path = cwd + "/images/cut.png"
    Img_Original = io.imread(path)
    if (len(np.shape(Img_Original)) == 3 and np.shape(Img_Original)[2] == 3):
        gray_image = cv2.cvtColor(Img_Original, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = Img_Original
    x, y, width, height = component = cv2.boundingRect(255-gray_image)
    Otsu_Threshold = threshold_otsu(gray_image)
    BW_Original = (gray_image < Otsu_Threshold) > 0
    # BW_Original = util.invert(BW_Original)
    io.imsave(cwd+"/debug/binary.png", img_as_uint(BW_Original))

    """ 
    ...................... Coarse Segmentation ................

    """
    averageStroke = int(SWTScrubber.scrub(path))
    kernel = np.ones((8,5), np.uint8) # ??????
    closingImg = cv2.morphologyEx(img_as_uint(BW_Original), cv2.MORPH_CLOSE, kernel)
    closingImg = closingImg > 0

    fineSegments, overSegments = get_segments(closingImg)
    [cv2.imwrite(cwd+"/debug/oversizedSegments_{}.png".format(i), convert_binary_to_normal_im(overSegments[i])) for i in range(len(overSegments))]
    [cv2.imwrite(cwd+"/debug/fineSegments_{}.png".format(i), convert_binary_to_normal_im(fineSegments[i])) for i in range(len(fineSegments))]

    for im in overSegments:
        # show_img(im)
        untouchingCases, touchingCases = get_segments(im, averageStroke)
        # [show_img(im) for im in touchingCases]
        # [show_img(im) for im in untouchingCases]

    """ 
    ................  Fine segmentation ...................

    """ 
    # Step 1: Thin each oversized segment
    # show_img(overSegments[2])
    size = np.shape(overSegments[2])
    AveWid = size[0]*0.8
    # print(AveWid)
    segment_skeleton = morphology.skeletonize(overSegments[1]).astype(np.uint16)
    # show_img(segment_skeleton)

    # Step 2: Draw vertical lines (VLs)
    h, w = shape = np.shape(segment_skeleton)
    # print(shape)
    # print("Num: {}".format(int(np.ceil(w/AveWid))))

    projection = [sum(segment_skeleton[:,i]) for i in range(w)]
    # print(projection)
    # plt.plot(projection)
    # plt.show()
    nMaximumPeaks = np.sort(projection)[w-int(np.ceil(w/AveWid)):w]
    # print(nMaximumPeaks)
    # maximumPeakPos = list(set([projection.index(i) for i in nMaximumPeaks]))
    maximumPeakPos = []
    for i in range(len(projection)):
        if projection[i] in nMaximumPeaks:
            maximumPeakPos.append(i)
    # print(maximumPeakPos)

    for point in maximumPeakPos:
        segment_skeleton[:, point] = 1
        segment_skeleton[int(h/2), point+1] = 1
        segment_skeleton[int(h/2), point-1] = 1
    # show_img(segment_skeleton)

    cv2.imwrite(cwd+"/debug/VL_image.png", convert_binary_to_normal_im(segment_skeleton))

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
    print(nodesOfVLs)

    result = find_bridges_between_VLs(segment_skeleton, graph, [25, 22])
    # # result = np.reshape(result, (len(result),1))
    # # print("Shape: ", np.shape(result))
    # x = [result[i][0] for i in range(len(result))]
    # # for i in range(len(result)):
    # #     x = result[i][0]
    # # print("AAAA:",x)
    # voronoi_points = np.array([get_node_coordinate(graph, r) for r in x])
    # print(voronoi_points)

    # Step 4: Seperated by Voronoi diagrams
    # vor = Voronoi(voronoi_points)
    # print("Points:{}".format(vor.points))
    # print("Ridge points: {}".format(vor.ridge_points))
    # print("Vertices:{}".format(vor.vertices))
    # print("Ridge vertices: {}".format(vor.ridge_vertices))
    # print("Regions: {}".format(vor.ridge_vertices))
    # print("Point region: {}".format(vor.point_region))
    # voronoi_plot_2d(vor)
    # plt.show()

    # voronoi_img = convert_binary_to_normal_im(segment_skeleton)
    # show_img(voronoi_img)
    # [cv2.circle(voronoi_img, (int(col), int(row)), 1, (255,0,0), -1) for row, col in vor.vertices]
    # show_img(voronoi_img)

    # Step 5: Reflect the segmentation boundaries to the original image
