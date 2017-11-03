import math
import numpy as np

def p6(edge_image, edge_thresh): # return [edge_thresh_image, hough_image]
    height = len(edge_image)
    width = len(edge_image[0])
    ret_img = [[0 for x in range(width)] for y in range(height)]
    rou_points = 500
    theta_points = 500

    for i in range(height):
        for j in range(width):
            if edge_image[i][j] < edge_thresh:
                continue
            else:
                ret_img[i][j] = 255
    rou_max = int(math.ceil(math.sqrt(width**2 + height**2)))
    # rou: from -rou_max to rou_max
    # thetas: form -pi/2 to pi/2
    rou_vector = np.linspace(-rou_max, rou_max, rou_points)
    thetas = np.linspace(-math.pi / 2, math.pi / 2, theta_points)

    # parameter space
    parameter_space = [[0 for x in range(len(thetas))] for y in range(2*rou_max)]
    for i in range(len(ret_img)):
        for j in range(len(ret_img[0])):
            if ret_img[i][j] == 0:
                continue
            for k in range(len(thetas)):
                rou = int(round(j*math.cos(thetas[k]) + i*math.sin(thetas[k])) + rou_max)
                parameter_space[rou][k] += 1
    # scale parameter space to range 0 ~ 255
    max_vote = 0
    m = len(parameter_space)
    n = len(parameter_space[0])
    for i in range(m):
        for j in range(n):
            k = parameter_space[i][j]
            if k > max_vote:
                max_vote = k
    for i in range(m):
        for j in range(n):
            parameter_space[i][j] = int(math.floor(255.0 * parameter_space[i][j] / max_vote))
    return ret_img, parameter_space

















