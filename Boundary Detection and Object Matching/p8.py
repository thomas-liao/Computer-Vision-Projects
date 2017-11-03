
from p7 import p7 as p7
import numpy as np
import cv2
import matplotlib.pyplot as plt


def p8(image, hough_image, edge_thresh_image, hough_thresh):
    p7_line_image = p7(image, hough_image, hough_thresh) # p7_line_image: line color: (200,0,0)


    # seems we need to "blur" the edge_thresh_image first to ensure better capture
    def convolution(input_1, input_2):  # input_1 is the larger matrix
        height = len(input_1)
        width = len(input_1[0])
        a = len(input_2)
        b = len(input_2[0])
        result = [[0 for x in range(width)] for y in range(height)]
        for i in range(height):
            for j in range(width):
                for k in range(a):
                    for l in range(b):
                        result[i][j] += input_2[k][l] * input_1[i - k][j - l]
        for i in range(height):
            result[i][0] = result[i][width - 1] = 0
        for j in range(width):
            result[0][j] = result[height-1][j] = 0
        return result
    filter = (1.0 / 3**2) * np.ones((3,3), dtype = np.uint8)

    blur_edge_image = convolution(edge_thresh_image, filter)




    # idea is: if line doesn't overlay with edge_thresh_image, remove it (recover pixel from image)
    height = len(image)
    width = len(image[0])
    for i in range(height):
        for j in range(width):
            if p7_line_image[i, j, 0] == 200 and p7_line_image[i,j,1] == 0 and p7_line_image[i,j,2] == 0:
                # if not overlay with edge_thresh_image, recover original pixel
                if (blur_edge_image[i][j] == 0):
                    p7_line_image[i,j,0] = p7_line_image[i,j,1] = p7_line_image[i,j,2] = image[i][j]


    return p7_line_image








