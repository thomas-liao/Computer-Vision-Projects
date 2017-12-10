import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import compute_affine_xform

def match_features_ratio(feature_coords1, descriptors1, feature_coords2, descriptors2):
    match = list()

    def SSD(m1, m2):
        length = len(m1)
        sum = 0
        for i in range(length):
            sum += (m1[i] - m2[i]) ** 2
        return sum

    for i in range(len(feature_coords1)):
        point1 = feature_coords1[i]
        m1 = descriptors1[point1[1], point1[0]]
        if (len(m1) == 0):
            continue
        SSD_result = dict()
        for j in range(len(feature_coords2)):
            point2 = feature_coords2[j]
            m2 = descriptors2[point2[1], point2[0]]
            if (len(m2) == 0):
                continue
            ssd1 = SSD(m1, m2)
            SSD_result[ssd1] = (i, j)

        keylist = SSD_result.keys()

        keylist.sort()
        if (keylist[0] / keylist[1] < 0.6):
            match.append((SSD_result[keylist[0]]))
    return match

