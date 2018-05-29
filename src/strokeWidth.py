# -*- encoding: utf-8 -*-
from __future__ import division
from collections import defaultdict
import math
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import scipy.sparse, scipy.spatial

diagnostics = True

class stroke_width_transform(object):
    @classmethod
    def scrub(cls, filepath):
        """
        Apply Stroke-Width Transform to image.
        """
        canny, sobelx, sobely, theta = cls._create_derivative(filepath)
        swt = cls._swt(theta, canny, sobelx, sobely)
        return swt

    @classmethod
    def _create_derivative(cls, filepath):
        img = cv2.imread(filepath,0)
        # img = ~ img
        # cv2.imwrite("invert_img.jpg", img)
        edges = cv2.Canny(img, 175, 320, apertureSize=3)
        # Create gradient map using Sobel
        sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=-1)
        sobely64f = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=-1)

        theta = np.arctan2(sobely64f, sobelx64f)
        if diagnostics:
            cv2.imwrite('/debug/edges.jpg',edges)
            cv2.imwrite('/debug/sobelx64f.jpg', np.absolute(sobelx64f))
            cv2.imwrite('/debug/sobely64f.jpg', np.absolute(sobely64f))
            # amplify theta for visual inspection
            theta_visible = (theta + np.pi)*255/(2*np.pi)
            # cv2.imwrite('theta.jpg', theta_visible)
        return (edges, sobelx64f, sobely64f, theta)

    @classmethod
    def _swt(self, theta, edges, sobelx64f, sobely64f):
        # create empty image, initialized to infinity
        swt = np.empty(theta.shape)
        # swt[:] = np.Infinity
        rays = []
        step_x_g = -1 * sobelx64f
        step_y_g = -1 * sobely64f
        mag_g = np.sqrt( step_x_g * step_x_g + step_y_g * step_y_g )
        # print(mag_g)
        cv2.imwrite("/debug/mag_g.jpg", np.absolute(mag_g))
        # show_img(mag_g)
        # np.seterr(divide='ignore', invalid='ignore')
        np.seterr(divide='ignore', invalid='ignore')

        grad_x_g = step_x_g / mag_g
        grad_x_g[np.isnan(grad_x_g)] = 0
        # print(grad_x_g)
        # show_img(grad_x_g)

        grad_y_g = step_y_g / mag_g
        grad_y_g[np.isnan(grad_y_g)] = 0
        # print(grad_y_g)
        # show_img(grad_y_g)

        for x in range(edges.shape[1]):
            for y in range(edges.shape[0]):
                if edges[y, x] > 0:
                    step_x = step_x_g[y, x]
                    step_y = step_y_g[y, x]
                    mag = mag_g[y, x]
                    grad_x = grad_x_g[y, x]
                    grad_y = grad_y_g[y, x]
                    ray = []
                    ray.append((x, y))
                    prev_x, prev_y, i = x, y, 0
                    while True:
                        i += 1
                        cur_x = math.floor(x + grad_x * i)
                        cur_y = math.floor(y + grad_y * i)

                        if cur_x != prev_x or cur_y != prev_y:
                            # we have moved to the next pixel!
                            try:
                                if edges[cur_y, cur_x] > 0:
                                    # found edge,
                                    ray.append((cur_x, cur_y))
                                    # theta_point = theta[y, x]
                                    # alpha = theta[cur_y, cur_x]
                                    thickness = math.sqrt((cur_x - x) * (cur_x - x) + (cur_y - y) * (cur_y - y))
                                    print("Thickness: {}".format(thickness))
                                    for (rp_x, rp_y) in ray:
                                        swt[rp_y, rp_x] = min(thickness, swt[rp_y, rp_x])
                                    if math.acos(grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]) < np.pi/2.0:
                                    # if math.acos(grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]) > 0:
                                        thickness = math.sqrt( (cur_x - x) * (cur_x - x) + (cur_y - y) * (cur_y - y) )
                                        print("Thickness: {}".format(thickness))
                                        for (rp_x, rp_y) in ray:
                                            swt[rp_y, rp_x] = min(thickness, swt[rp_y, rp_x])
                                        rays.append(ray)
                                    break
                                # this is positioned at end to ensure we don't add a point beyond image boundary
                                ray.append((cur_x, cur_y))
                            except IndexError:
                                # reached image boundary
                                break
                            prev_x = cur_x
                            prev_y = cur_y

        # Compute median SWT
        # print("Ray: {}".format(ray))
        # print("SWT: {}".format(swt))
        for ray in rays:
            median = np.max([swt[y, x] for (x, y) in ray])
            # print(median)
            for (x, y) in ray:
                swt[y, x] = min(median, swt[y, x])
        # if diagnostics:
        #     cv2.imwrite('swt.jpg', swt * 100)

        # print("Stroke-Width: {}".format(np.mean(swt/10)))
        return np.ceil(np.mean(swt)/30)

    @classmethod
    def _connect_components(cls, swt):
        # STEP: Compute distinct connected components
        # Implementation of disjoint-set
        class Label(object):
            def __init__(self, value):
                self.value = value
                self.parent = self
                self.rank = 0
            def __eq__(self, other):
                if type(other) is type(self):
                    return self.value == other.value
                else:
                    return False
            def __ne__(self, other):
                return not self.__eq__(other)

        ld = {}

        def MakeSet(x):
            try:
                return ld[x]
            except KeyError:
                item = Label(x)
                ld[x] = item
                return item

        def Find(item):
            # item = ld[x]
            if item.parent != item:
                item.parent = Find(item.parent)
            return item.parent

        def Union(x, y):
            """
            :param x:
            :param y:
            :return: root node of new union tree
            """
            x_root = Find(x)
            y_root = Find(y)
            if x_root == y_root:
                return x_root

            if x_root.rank < y_root.rank:
                x_root.parent = y_root
                return y_root
            elif x_root.rank > y_root.rank:
                y_root.parent = x_root
                return x_root
            else:
                y_root.parent = x_root
                x_root.rank += 1
                return x_root

        # apply Connected Component algorithm, comparing SWT values.
        # components with a SWT ratio less extreme than 1:3 are assumed to be
        # connected. Apply twice, once for each ray direction/orientation, to
        # allow for dark-on-light and light-on-dark texts
        trees = {}
        # Assumption: we'll never have more than 65535-1 unique components
        label_map = np.zeros(shape=swt.shape, dtype=np.uint16)
        next_label = 1
        # First Pass, raster scan-style
        swt_ratio_threshold = 3.0
        for y in range(swt.shape[0]):
            for x in range(swt.shape[1]):
                sw_point = swt[y, x]
                if sw_point < np.Infinity and sw_point > 0:
                    neighbors = [(y, x-1),   # west
                                 (y-1, x-1), # northwest
                                 (y-1, x),   # north
                                 (y-1, x+1)] # northeast
                    connected_neighbors = None
                    neighborvals = []

                    for neighbor in neighbors:
                        # west
                        try:
                            sw_n = swt[neighbor]
                            label_n = label_map[neighbor]
                        except IndexError:
                            continue
                        if label_n > 0 and sw_n / sw_point < swt_ratio_threshold and sw_point / sw_n < swt_ratio_threshold:
                            neighborvals.append(label_n)
                            if connected_neighbors:
                                connected_neighbors = Union(connected_neighbors, MakeSet(label_n))
                            else:
                                connected_neighbors = MakeSet(label_n)

                    if not connected_neighbors:
                        # We don't see any connections to North/West
                        trees[next_label] = (MakeSet(next_label))
                        label_map[y, x] = next_label
                        next_label += 1
                    else:
                        # We have at least one connection to North/West
                        label_map[y, x] = min(neighborvals)
                        # For each neighbor, make note that their respective connected_neighbors are connected
                        # for label in connected_neighbors. @todo: do I need to loop at all neighbor trees?
                        trees[connected_neighbors.value] = Union(trees[connected_neighbors.value], connected_neighbors)

        # Second pass. re-base all labeling with representative label for each connected tree
        layers = {}
        contours = defaultdict(list)
        for x in range(swt.shape[1]):
            for y in range(swt.shape[0]):
                if label_map[y, x] > 0:
                    item = ld[label_map[y, x]]
                    common_label = Find(item).value
                    label_map[y, x] = common_label
                    contours[common_label].append([x, y])
                    try:
                        layer = layers[common_label]
                    except KeyError:
                        layers[common_label] = np.zeros(shape=swt.shape, dtype=np.uint16)
                        layer = layers[common_label]

                    layer[y, x] = 1
        return layers

def show_img(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

# file = os.getcwd() + "/images/cut.png"
# stroke_width = SWTScrubber.scrub(file)





