import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def match_features(feature_coords1, feature_coords2, image1, image2):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        feature_coords1 (list of tuples): list of (row,col) tuple feature coordinates from image1
        feature_coords2 (list of tuples): list of (row,col) tuple feature coordinates from image2
        image1 (numpy.ndarray): The input image corresponding to features_coords1
        image2 (numpy.ndarray): The input image corresponding to features_coords2
    Returns:
        matches (list of tuples): list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 0-th feature
                                  in feature_coords2 are determined to be matches, the list should contain (4,0).
    """
    match = list()

    image1_color = image1
    image2_color = image2
    # convert image to gray scale
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    len_f1 = len(feature_coords1)
    len_f2 = len(feature_coords2)
    print "features_coods1 is"
    print feature_coords1
    print "features_coor2 ise"
    print feature_coords2
    def unit_variance_matrix(image, tuple, depth):
        i0 = tuple[0] - depth
        j0 = tuple[1] - depth
        N = (depth*2 + 1) ** 2
        window = np.zeros((depth*2 + 1, depth*2 + 1), dtype = np.float64)
        sum = 0
        for i in range(2*depth + 1):
            for j in range(2*depth + 1):
                window[i][j] = image[i + i0][j + j0]
                sum += window[i][j]
        miu = 1.0 * sum / N
        sigma_sqr = 0.0
        for i in range(2*depth + 1):
            for j in range(2*depth + 1):
                window[i][j] -= miu
                sigma_sqr += window[i][j]**2
        sigma_sqr = 1.0 * sigma_sqr / N
        sigma = math.sqrt(sigma_sqr)
        window = window / sigma # now window is ZN(x, y) matrix
        return window

    def cc1(window1, window2):
        height = len(window1)
        width = len(window1[0])
        N = height * width
        sum = 0.0
        for i in range(height):
            for j in range(width):
                sum += window1[i][j] * window2[i][j]
        return 1.0 * sum / N

    def cc2(tuple1, tuple2, img1, img2, depth): # just a function to wrap everything up so we can use it easily
        if (tuple1[0] < depth or tuple2[0] < depth or tuple1[0] > len(img1) - depth-10 or tuple2[0] > len(img2) - depth-10 or
            tuple1[1] < depth or tuple2[1] < depth or tuple1[1] > len(img1[0]) - depth-10 or tuple2[1] > len(img1[0]) - depth-10) or (tuple1[1] < depth or tuple2[1] < depth or tuple1[1] > len(img1) - depth - 10 or tuple2[1] > len(
        img2) - depth - 10 or
    tuple1[0] < depth or tuple2[0] < depth or tuple1[0] > len(img1[0]) - depth - 10 or tuple2[0] > len(
        img1[0]) - depth - 10) :
            return -2 # out of boundary, no cc!
        window1 = unit_variance_matrix(img1, tuple1, depth)
        window2 = unit_variance_matrix(img2, tuple2, depth)
        return cc1(window1, window2)

    def match_two(tuple, cood, img1, img2, depth): # return the index of best match in cood2 for a tuple point (from cood1)
        length = len(cood)
        record = -1 # record for cc
        matching_index = -1
        for i in range(length):
            c = cc2(tuple, cood[i], img1, img2, depth)
            if c > record:
                record = c
                matching_index = i
        if record < 0.2: # if record is too small, then still return -1 indicating nothing found
            return -1
        return matching_index



    res1 = list()
    res2 = list()

    for i in range(len_f1):
        res1.append(match_two(feature_coords1[i], feature_coords2, image1, image2, 7))
    for i in range(len_f2):
        res2.append(match_two(feature_coords2[i], feature_coords1, image2, image1, 7))

    print "coord 1 matching to coord2 is"
    print res1
    print "coodr2 matching to coord1 is"
    print res2
    for i in range(len_f1):
        if (res2[res1[i]] == i):
            match.append((i, res1[i]))

    # draw side-by-side image with matched features
    draw = draw_sbs(image1_color, image2_color, match, feature_coords1, feature_coords2)
    cv2.imshow("match_features_img1_img2", draw)
    # cv2.imwrite("math_features_img1_img2.png", draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return match


def sbs(image_a, image_b): #image_a, image_b: color image
    ha, wa = image_a.shape[0:2]
    hb, wb = image_b.shape[0:2]
    max_height = np.max([ha, hb])
    max_width = wa + wb
    ret_img = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    ret_img[0:ha, 0:wa] = image_a
    ret_img[0:hb, wa:wa + wb] = image_b
    return ret_img

def draw_sbs(image_a, image_b, match, feature_coods1, feature_coods2):
    combined_img = sbs(image_a, image_b)
    wa = image_a.shape[1]
    for i in range(len(match)):
        point_a = feature_coods1[match[i][0]]
        point_b = feature_coods2[match[i][1]]
        print point_a
        print point_b
        cv2.circle(combined_img, (point_a[1], point_a[0]), 3, (0,0,255), -1 );
        cv2.circle(combined_img, (point_b[1] + wa, point_b[0]), 3, (0, 0, 255), -1);
        cv2.line(combined_img, (point_a[1], point_a[0]), (point_b[1] + wa, point_b[0]), (0, 255, 0), 1)
    return combined_img




