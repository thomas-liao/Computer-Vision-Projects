import math
import cv2
import numpy as np

def p7(image, hough_image, hough_thresh): # return line_image

    m = len(hough_image)
    n = len(hough_image[0])
    pi = math.pi
    height = len(image)
    width = len(image[0])
    rou_max = int(math.ceil(math.sqrt(height ** 2 + width ** 2)))
    unit_theta = math.pi / n
    unit_rou = 2*rou_max / m

    # convert image to 3 channel (three_channel_img)
    three_channel_img = np.zeros((height, width, 3), dtype = np.uint8)
    for i in range(height):
        for j in range(width):
            three_channel_img[i,j,0] = three_channel_img[i, j, 1] = three_channel_img[i, j, 2] = image[i][j]

    hough_image_threshold = [[0 for x in range(n)] for y in range(m)]

    for i in range(m):
        for j in range(n):
            if hough_image[i][j] > hough_thresh:
                hough_image_threshold[i][j] = hough_image[i][j]

    # now we have hough_image_threshold, use it to draw
##################################################################################################################################################
    # brief explanation on drawing: first, xo, yo: the nearest point (to origin) on the line, which is rou*cos(theta) and rou*sin(theta) respectively
    # suppose a point move away (2 directions) from (xo, yo), then if it moves toward x axis, x1 = xo + sin(theta) and y1 = y0 - cos(theta)
    # on the contrary, if the points moves toward y, then y2 = y0 + cos(theta) and x2 = x0 - sin(theta), that is how we got point (x1, y1)
    # and point (x2, y2). now what we really need to do is to set this "movement" value extremely large to make sure it goes beyong the
    # boundary of image so we can get the line crossing the whoe image. Here I choose 9999 (moving toward x axis) and -9999 (moving toward y axis)
    # as long as the movement value is large enough, we should get the line crossing the image as we desired
##################################################################################################################################################
    for i in range(m):
        for j in range(n):
            if hough_image_threshold[i][j] == 0:
                continue
            theta = -pi/2 + j * unit_theta
            rou = -rou_max + i * unit_rou
            a = np.cos(theta)
            b = np.sin(theta)
            xo = a * rou
            yo = b * rou
            x1 = int(xo + 9999 * (-b))
            y1 = int(yo + 9999 * a)
            x2 = int(xo - 9999 * (-b))
            y2 = int(yo - 9999 * a)
            cv2.line(three_channel_img, (x1, y1), (x2, y2), (200, 0, 0), 1)
            # see above explanation on drawing
    return three_channel_img






