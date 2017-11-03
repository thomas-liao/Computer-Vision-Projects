from p3 import p3 as p3
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

def p4(labels_in, database_in): # return overlays_out
    database_in_2, img = p3(labels_in)
    overlay_label = [] # overlayed label in database_in_2
    for label_2 in database_in_2:
        roundness_2 = database_in_2[label_2]["roundness"]
        for label_1 in database_in:
            roundness_1 = database_in[label_1]["roundness"]
            if abs(roundness_1 - roundness_2) / roundness_1 < 0.1: # criteria: 90% "resemblance"
                overlay_label.append(label_2)


    # now that we have labels (for overlay), we can draw


    # convert 1 channel imgage to 3 channel for cv2 for drawing
    height = len(labels_in)
    width = len(labels_in[0])
    draw_img = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if labels_in[i][j] != 0:
                draw_img[i, j, 0] = draw_img[i, j, 1] = draw_img[i, j, 2] = 255

    # draw
    delta = 80
    for lb in overlay_label:
        theta = database_in_2[lb]["orientation_theta"]
        x1 = int(database_in_2[lb]["x"])
        y1 = int(database_in_2[lb]["y"])
        x2 = int(x1 + delta * math.cos(theta))
        y2 = int(y1 + delta * math.sin(theta))
        cv2.line(draw_img, (x1, y1), (x2, y2),(255, 0 ,0), 2)
        cv2.circle(draw_img, (x1, y1), 5, (0, 255, 0), -1)


    return draw_img














