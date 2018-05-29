import matplotlib.pyplot as plt
from skimage import util, img_as_uint, io, morphology
import numpy as np
import cv2
from matplotlib import pyplot
import skimage.color as color
from skimage.filters import threshold_otsu
import copy

def convert_binary_to_normal_im(binary):
    gray_img = ((1-binary)*255).astype('uint8')
    return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

def show_img(img, winname="TEST"):
    pyplot.figure()
    pyplot.imshow(img)
    pyplot.title(winname)
    pyplot.show()

def getSeperablePoints(list,up=0,down=1):
    result = []
    for i in range(down, len(list)):
        if ((list[i]-1) != list[i-1]):
            up = i-1
            result.append(int(np.mean(list[down:up])))
            down = i
        elif (i == (len(list)-1)):
            result.append(int(np.mean(list[down:i])))
    return result

if __name__ == '__main__':
    path = "/home/thanh/Desktop/segmentation/images/cut.png"
    Img_Original = io.imread(path)
    # show_img(Img_Original)
    if (len(np.shape(Img_Original)) == 3 and np.shape(Img_Original)[2] == 3):
        gray_image = cv2.cvtColor(Img_Original, cv2.COLOR_RGB2GRAY)
    x, y, width, height = component = cv2.boundingRect(255-Img_Original)
    # print(component)
    # rgb_image = cv2.cvtColor(Img_Original, cv2.COLOR_GRAY2BGR)
    # Img_Original = color.rgb2gray(Img_Original)
    # plt.imshow(Img_Original, cmap=plt.cm.gray)
    # plt.show()
    Otsu_Threshold = threshold_otsu(Img_Original)
    BW_Original = (Img_Original < Otsu_Threshold) > 0
    # BW_Original = util.invert(BW_Original)
    io.imsave("binary.png", img_as_uint(BW_Original))

    # ....... Step 1: Estimation of the average width (AvWid)
    # the average width of characters is 80,
    # theta (0.9) is the coefficient that is determined from the sample text lines
    theta = 0.9
    AveWid = height*theta
    maxWid = AveWid*1.5
    # print(AveWid)
    kernel = np.ones((8,5), np.uint8)
    # print(kernel)
    # print(BW_Original*1)
    # im = io.imread("/home/thanh/Desktop/segmentation/binary.png")

    # ........ Step 2: Estimation of the average stroke width (AW)
    # swt = swt.strokeWidthTransform(BW_Original, 1)
    # plt.subplot(3,2,3)
    # plt.imshow(swt, cmap="gray", interpolation="none")
    # plt.title('positive swt of image')
    # plt.show()

    # ........ Step 3 Smearing
    closingImg = cv2.morphologyEx(img_as_uint(BW_Original), cv2.MORPH_CLOSE, kernel)
    closingImg = closingImg > 0
    verticalProjection = [sum(closingImg.transpose()[i][:]) for i in range(width)]
    # plt.plot(verticalProjection)
    # plt.show()
    # ......... Step 4: Zero point segmentation
    points = [i for i in range(len(verticalProjection)) if verticalProjection[i] == 0]
    # print(points)
    seperablePoints = getSeperablePoints(points)
    # print(seperablePoints)
    im_out = convert_binary_to_normal_im(closingImg)
    # [cv2.circle(im_out, (col, height-1), 4, (255,0,0), -1) for col in seperablePoints]
    [cv2.line(im_out, (col,0), (col, height-1), (255,0,0), 4, -1) for col in seperablePoints]
    cv2.imwrite("sperablePoints.png", im_out)
    # show_img(im_out)

    # ......... Step 5: segmentation by AW
    fineSegmentPoints = [[seperablePoints[i-1], seperablePoints[i]] for i in range(1, len(seperablePoints)) if ((seperablePoints[i] - seperablePoints[i-1]) < maxWid)]
    oversizedSegmentPoints = [[seperablePoints[i-1], seperablePoints[i]] for i in range(1, len(seperablePoints)) if ((seperablePoints[i] - seperablePoints[i-1]) > maxWid)]

    oversizedSegments = [closingImg[0:width, i:j] for i,j in oversizedSegmentPoints]
    # [show_img(i) for i in oversizedSegments]
    [cv2.imwrite("oversizedSegments_{}.png".format(i), convert_binary_to_normal_im(oversizedSegments[i])) for i in range(len(oversizedSegments))]

    fineSegments = [closingImg[0:width, i:j] for i,j in fineSegmentPoints]
    [show_img(i) for i in fineSegments]
    # [cv2.imwrite("fineSegments_{}.png".format(i), convert_binary_to_normal_im(oversizedSegments[i])) for i in range(len(oversizedSegments))]

    # io.imsave("closing.png", closingImg)

    segment_skeleton = morphology.skeletonize(oversizedSegments[0]).astype(np.uint16)

    # show_img(segment_skeleton)
    # normal_im = convert_binary_to_normal_im(BW_Skeleton)
    # color_image = cv2.cvtColor(BW_Skeleton, cv2.COLOR_GRAY2BGR)
    # io.imsave("rgb_skeleton.png", normal_im)
    height, width = shape = np.shape(segment_skeleton)
    print(width, height)
    # print("properties: {}".format(np.ceil(width/AveWid)), width, AveWid)
    projection = [sum(segment_skeleton.transpose()[i][:]) for i in range(width)]
    plt.plot(projection)
    plt.show()
    tmp = np.sort(projection)[(width-int(np.ceil(width/AveWid))):width]
    print(tmp)
    x = [projection.index(i) for i in tmp]
    print(x)
    VL_img = convert_binary_to_normal_im(segment_skeleton)
    [cv2.line(VL_img, (col,0), (col,height-1), (255,0,0), 4, -1) for col in x]
    cv2.imwrite("VL_img.png", VL_img)
    show_img(VL_img)
