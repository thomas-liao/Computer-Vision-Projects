# Author: TK
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_proj_xform(matches, features1, features2, image1, image2):
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
        proj_xform (numpy.ndarray): a 3x3 Projective transformation matrix between the two images, computed using the matches.
    """

    proj_xform = np.zeros((3, 3))
    def calculate_h (matches, features1, features2, random_index):
        a = np.zeros((8,9), dtype = np.float64)
        for i in range(4):
            point1 = features1[matches[random_index[i]][0]]
            point2 = features2[matches[random_index[i]][1]]
            a[2*i] = [point1[1], point1[0], 1, 0, 0, 0, -point2[1]*point1[1], -point2[1]*point1[0], -point2[1]]
            a[2*i+1] = [0, 0, 0, point1[1],point1[0], 1, -point2[0]*point1[1], - point2[0]*point1[0], -point2[0]]
        _,_,ei = np.linalg.svd(a)
        min_vec = ei[-1]
        min_vec = min_vec / min_vec[-1]
        min_vec = np.reshape(min_vec, (3,3))
        return min_vec

    def ransac(N, matches, features1, features2): # return h with max voting
        h_record = np.zeros((N,3,3), dtype = np.float64)
        vote_record = np.zeros((N,1), dtype = np.uint64)
        inlier = list()
        for i in range(N):
            inlier.append([])
            random_index = np.random.random_integers(0, len(matches)-1, 4)
            h_record[i] = calculate_h(matches, features1, features2, random_index)
            # then calculate vote_record
            for j in range(len(matches)):
                h = h_record[i]
                point1 = features1[matches[j][0]]
                point2 = features2[matches[j][1]]
                temp = np.array([point1[1], point1[0], 1])
                prime = np.dot(temp, h)
                if np.sqrt((prime[1] - point2[0])**2 + (prime[0] - point2[1])**2) < 10:
                    vote_record[i] += 1
                    inlier[i].append(j)
        # iterate through vote_record to get max
        max_index = -1
        max_vote = 0
        for i in range(9999):
            if vote_record[i] > max_vote:
                max_vote = vote_record[i]
                max_index = i
        h = h_record[max_index]
        return h, inlier[max_index]


    def sbs(image_a, image_b):  # image_a, image_b: color image
        ha, wa = image_a.shape[0:2]
        hb, wb = image_b.shape[0:2]
        max_height = np.max([ha, hb])
        max_width = wa + wb
        ret_img = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        ret_img[0:ha, 0:wa] = image_a
        ret_img[0:hb, wa:wa + wb] = image_b
        print "shape of ret_img"
        print ret_img.shape
        print image_a.shape
        return ret_img

    def draw_sbs_inlier_and_outlier(image_a, image_b, match, features1, features2, inlier):
        combined_img = sbs(image_a, image_b)
        wa = image_a.shape[1]
        for i in range(len(match)):
            point_a = features1[match[i][0]]
            point_b = features2[match[i][1]]
            print point_a
            print point_b
            cv2.circle(combined_img, (point_a[1], point_a[0]), 3, (0, 0, 255), -1);
            cv2.circle(combined_img, (point_b[1] + wa, point_b[0]), 3, (0, 0, 255), -1);
            if(i in inlier):
                cv2.line(combined_img, (point_a[1], point_a[0]), (point_b[1] + wa, point_b[0]), (0, 255, 0), 1)
            else:
                cv2.line(combined_img, (point_a[1], point_a[0]), (point_b[1] + wa, point_b[0]), (0, 0, 255), 1)
        return combined_img


    proj_xform, inlier = ransac(9999, matches, features1, features2)



    draw1 = draw_sbs_inlier_and_outlier(image1, image2, matches, features1,features2, inlier)

    # draw inlier_outlier image
    draw1 = draw_sbs_inlier_and_outlier(image1, image2, matches, features1, features2, inlier)
    cv2.imshow("proj_match_features_inlier_outlier_img1_img2", draw1)
    # cv2.imwrite("proj_matches_inlier outlier_img1_img2.png", draw1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # draw overlay image
    height, width = image2.shape[:2]

    warp_image = cv2.warpPerspective(image1, proj_xform, (width, height))
    draw2 = cv2.addWeighted(warp_image, 0.5, image2, 0.5, 1)
    cv2.imshow('proj_overlay_img1_img2', draw2)
    # cv2.imwrite('proj_overlay_wall1_wall3.png', draw2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




    return proj_xform