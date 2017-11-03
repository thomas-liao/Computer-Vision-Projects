import math
import numpy as np
import cv2

import matplotlib.pyplot as plt




def p3(labeled_img): #return [database_out, overlays_out]
    # get labels a a list
    height = len(labeled_img)
    width = len(labeled_img[0])
    labels = []

    def e_function(a, b, c, theta):
        return a * math.sin(theta)**2 - b * math.sin(theta) * math.cos(theta) + c * math.cos(theta) ** 2

    for i in range(height):
        for j in range(width):
            if labeled_img[i][j] == 0 or labeled_img[i][j] in labels:
                continue
            labels.append(labeled_img[i][j])
    i_mean = dict()
    j_mean = dict()
    a_prime = dict()
    b_prime = dict()
    c_prime = dict()
    a = dict()
    b = dict()
    c = dict()
    area = dict()
    theta = dict()
    theta2 = dict()
    e_min = dict()
    e_max = dict()
    roundness = dict()

    for n in labels:
        i_mean[n] = 0
        j_mean[n] = 0
        a_prime[n] = 0
        b_prime[n] = 0
        c_prime[n] = 0
        area[n] = 0
        a[n] = 0
        b[n] = 0
        c[n] = 0
        theta2[n] = 0
        e_min[n] = 0
        e_max[n] = 0

    for i in range(height):
        for j in range(width):
            n = labeled_img[i][j]
            if n == 0:
                continue
            area[n] += 1
            i_mean[n] += i
            j_mean[n] += j
            a_prime[n] += j * j
            b_prime[n] += 2 * i * j
            c_prime[n] += i * i

    # calculate i_mean and j_mean
    for label in labels:
        i_mean[label] = 1.0 * i_mean[label] / area[label]
        j_mean[label] = 1.0 * j_mean[label] / area[label]
        a[label] = a_prime[label] - 1.0 * area[label] * j_mean[label] ** 2
        c[label] = c_prime[label] - 1.0 * area[label] * i_mean[label] ** 2
        b[label] = b_prime[label] - 2.0 * area[label] * i_mean[label] * j_mean[label]
        theta[label] = 0.5 * math.atan2(b[label], a[label] - c[label])
        theta2[label] = theta[label] + 3.1415926 / 2.0
        e_min[label] = e_function(a[label], b[label], c[label], theta[label])
        e_max[label] = e_function(a[label], b[label], c[label], theta2[label])
        roundness[label] = e_min[label] / e_max[label]

########################################################################################################################
    # from calculation, transform a_prime, b_prime, c_prime (w.r.t. origin) to a, b, c(w.r.t. object center)
    # i: i_mean, j: j_mean
    # a = a_prime - A*j*j
    # b = b_prime - 2*A*i*j
    # c = c_prime - A*i*i
########################################################################################################################
    # convert 1 channel imgage to 3 channel for cv2
    draw_img = np.zeros((height, width, 3), dtype = np.uint8)
    for i in range(height):
        for j in range(width):
            if labeled_img[i][j] != 0:
                draw_img[i,j,0] = draw_img[i,j,1] = draw_img[i,j,2] = 255

    # draw
    delta = 80
    for label in labels:
        x1 = int(j_mean[label])
        y1 = int(i_mean[label])
        x2 = int(x1 + delta * math.cos(theta[label]))
        y2 = int(y1 + delta * math.sin(theta[label]))
        cv2.line(draw_img, (x1, y1), (x2, y2), (255, 0 ,0), 2)
        cv2.circle(draw_img,(x1, y1),5, (0, 255, 0), -1)



    # create database_out and put data in it
    database_out = dict()
    # attributes "x","y","area","orientation_theta","roundness"
    for label in labels:
        database_out[label] = dict()
        database_out[label]["x"] = j_mean[label]
        database_out[label]["y"] = i_mean[label]
        database_out[label]["area"] = area[label]
        database_out[label]["orientation_theta"] = theta[label]
        database_out[label]["roundness"] = roundness[label]

    return database_out, draw_img





