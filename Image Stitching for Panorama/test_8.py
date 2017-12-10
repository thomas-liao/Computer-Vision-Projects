# Author: TK
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_affine_xform(matches, features1, features2, image1, image2):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        matches (list of tuples): list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 0-th feature
                                  in feature_coords2 are determined to be matches, the list should contain (4,0).
        features1 (list of tuples) : list of feature coordinates corresponding to image1
        features2 (list of tuples) : list of feature coordinates corresponding to image2
        image1 (numpy.ndarray): The input image corresponding to features_coords1
        image2 (numpy.ndarray): The input image corresponding to features_coords2
    Returns:
        affine_xform (numpy.ndarray): a 3x3 Affine transformation matrix between the two images, computed using the matches.
    """


    affine_xform = np.zeros((3, 3))





    def calculate_h (matches, features1, features2, random_index):
        a = np.zeros((8,9), dtype = np.float64)
        for i in range(3):
            point1 = features1[matches[random_index[i]][0]]
            point2 = features2[matches[random_index[i]][1]]
            a[2*i] = [point1[0], point1[1], 1, 0, 0, 0, -point2[0]*point1[0], -point2[0]*point1[1], -point2[0]]
            a[2*i+1] = [0, 0, 0, point1[0],point1[1], 1, -point2[1]*point1[0], - point2[1]*point1[1], -point2[1]]
        _,_,ei = np.linalg.svd(a)
        min_vec = ei[-1]
        min_vec = min_vec / min_vec[-1]
        return min_vec

    def ransac(N, matches, features1, features2): # return h with max voting
        h_record = np.zeros((N, 9), dtype = np.float64)
        vote_record = np.zeros(N, dtype = np.uint64)

        for i in range(N):
            random_index = np.random.random_integers(0, len(matches)-1, 3)
            h_record[i] = calculate_h(matches, features1, features2, random_index)
            # then calculate vote_record
            for j in range(len(matches)):
                h = np.reshape(h_record[i], (3,3))
                prime = np.dot(h, np.array([features1[j][0], features1[j][1], 1]))
                if np.sqrt((prime[0] - features2[j][0])**2 + (prime[1] - features2[j][1])**2) < 7:
                    vote_record[i] += 1
        # iterate through vote_record to get max
        max = 0
        max_index = -1
        for i in range(N):
            if vote_record[i] > max:
                max = vote_record[i]
                max_index = i
        return h_record[max_index]

    affine_xform = np.reshape(ransac(9999, matches, features1, features2), (3,3))

    return affine_xform


