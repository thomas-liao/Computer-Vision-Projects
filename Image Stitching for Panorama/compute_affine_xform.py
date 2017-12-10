# Author: TK
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_affine_xform(matches, features1, features2, image1, image2):
    affine_xform = np.zeros((3, 3))

    def calculate_h (matches, features1, features2, random_index):
        a = np.zeros((6,6), dtype = np.float64)
        b = np.zeros((6,1), dtype = np.float64)
        for i in range(3):
            point1 = features1[matches[random_index[i]][0]]
            point2 = features2[matches[random_index[i]][1]]
            a[2*i] = [point1[1], point1[0], 1, 0, 0, 0]
            a[2*i+1] = [0, 0, 0, point1[1],point1[0], 1]
            b[2*i] = point2[1]
            b[2*i+1] = point2[0]
        # calculate t
        last = np.array([0, 0, 1])
        t = np.dot(calculate_psedo_inverse(a), b)
        r = np.reshape(t, (2, 3))
        r = np.vstack((r, last))
        return r

    def calculate_psedo_inverse(a):
        ret = np.linalg.pinv(a)
        # ret = np.dot(np.linalg.inv(np.dot(a.T, a)), a.T) # use this one
        return ret


    def ransac(N, matches, features1, features2): # return h with max voting
        h_record = np.zeros((N, 3,3), dtype = np.float64)
        vote_record = np.zeros((N,1), dtype = np.uint64)
        inlier = list()
        for i in range(N):
            inlier.append([])
            random_index = np.random.random_integers(0, len(matches)-1, 3)
            while (len(set(random_index)) < 3):
                random_index = np.random.random_integers(0, len(matches) - 1, 3)

            h_record[i] = calculate_h(matches, features1, features2, random_index)
            # then calculate vote_record
            for j in range(len(matches)):
                h = h_record[i]
                point1 = features1[matches[j][0]]
                point2 = features2[matches[j][1]]
                temp = np.array([point1[1], point1[0], 1])
                prime = np.dot(temp, h)
                if np.sqrt((prime[1] - point2[0])**2 + (prime[0] - point2[1])**2) < 15:
                    vote_record[i] += 1
                    inlier[i].append(j)
        # iterate through vote_record to get max

        max_index = -1
        max_vote = 0
        for i in range(9999):
            if vote_record[i] > max_vote:
                max_vote = vote_record[i]
                max_index = i
        return h_record[max_index], inlier[max_index]


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


    affine_xform, inlier = ransac(9999, matches, features1, features2)

    # average inlier results to get better result
    if len(inlier) > 3:
        h_record = np.zeros((1000, 3, 3), dtype=np.float64)
        for i in range(1000):
            random_index = np.random.random_integers(0, len(inlier)-1, 3)
            while (len(set(random_index)) < 3):
                random_index = np.random.random_integers(0, len(inlier) - 1, 3)
            a = np.zeros((6, 6), dtype=np.float64)
            b = np.zeros((6, 1), dtype=np.float64)
            for j in range(3):
                point1 = features1[matches[inlier[random_index[j]]][0]]
                point2 = features2[matches[inlier[random_index[j]]][1]]
                a[2 * j] = [point1[1], point1[0], 1, 0, 0, 0]
                a[2 * j + 1] = [0, 0, 0, point1[1], point1[0], 1]
                b[2 * j] = point2[1]
                b[2 * j + 1] = point2[0]
            last = np.array([0, 0, 1])
            t = np.dot(calculate_psedo_inverse(a), b)
            r = np.reshape(t, (2, 3))
            r = np.vstack((r, last))
            h_record[i] = r

        h = np.mean(h_record, axis = 0)
        affine_xform = h


    # draw inlier_outlier image
    draw1 = draw_sbs_inlier_and_outlier(image1, image2, matches, features1,features2, inlier)

    cv2.imshow("affine_match_features_inlier_outlier_img1_img2", draw1)
    # cv2.imwrite("affine_matches_inlier outlier_img1_img2.png", draw1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # draw overlay image
    height, width = image2.shape[:2]

    warp_image = cv2.warpPerspective(image1, affine_xform, (width, height))
    draw2 = cv2.addWeighted(warp_image, 0.5, image2, 0.5, 1)
    cv2.imshow('affine_overlay_wall1_wall2', draw2)
    cv2.imwrite('affine_overlay_wall1_wall2.png', draw2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return affine_xform


# note: codes from line 143 to end are identical to above code from line 1 - 140, only difference is the name of images saved... couldn't figure out a better way
# of doing it so I just copy as pasted and change the name.... which is not an elegant solution, but works.
def compute_affine_xform_for_ssift(matches, features1, features2, image1, image2):
    affine_xform = np.zeros((3, 3))

    def calculate_h (matches, features1, features2, random_index):
        a = np.zeros((6,6), dtype = np.float64)
        b = np.zeros((6,1), dtype = np.float64)
        for i in range(3):
            point1 = features1[matches[random_index[i]][0]]
            point2 = features2[matches[random_index[i]][1]]
            a[2*i] = [point1[1], point1[0], 1, 0, 0, 0]
            a[2*i+1] = [0, 0, 0, point1[1],point1[0], 1]
            b[2*i] = point2[1]
            b[2*i+1] = point2[0]
        # calculate t
        last = np.array([0, 0, 1])
        t = np.dot(calculate_psedo_inverse(a), b)
        r = np.reshape(t, (2, 3))
        r = np.vstack((r, last))
        return r

    def calculate_psedo_inverse(a):
        ret = np.dot(np.linalg.inv(np.dot(a.T, a)), a.T) # use this one
        return ret


    def ransac(N, matches, features1, features2): # return h with max voting
        h_record = np.zeros((N, 3,3), dtype = np.float64)
        vote_record = np.zeros((N,1), dtype = np.uint64)
        inlier = list()
        for i in range(N):
            inlier.append([])
            random_index = np.random.random_integers(0, len(matches)-1, 3)
            while (len(set(random_index)) < 3):
                if (len(matches) < 3):
                    break
                random_index = np.random.random_integers(0, len(matches) - 1, 3)

            h_record[i] = calculate_h(matches, features1, features2, random_index)
            # then calculate vote_record
            for j in range(len(matches)):
                h = h_record[i]
                point1 = features1[matches[j][0]]
                point2 = features2[matches[j][1]]
                temp = np.array([point1[1], point1[0], 1])
                prime = np.dot(temp, h)
                if np.sqrt((prime[1] - point2[0])**2 + (prime[0] - point2[1])**2) < 15:
                    vote_record[i] += 1
                    inlier[i].append(j)
        # iterate through vote_record to get max

        max_index = -1
        max_vote = 0
        for i in range(9999):
            if vote_record[i] > max_vote:
                max_vote = vote_record[i]
                max_index = i
        return h_record[max_index], inlier[max_index]


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


    affine_xform, inlier = ransac(9999, matches, features1, features2)

    # # average inlier results to get better result
    # if len(inlier) > 3:
    #     h_record = np.zeros((1000, 3, 3), dtype=np.float64)
    #     for i in range(1000):
    #         random_index = np.random.random_integers(0, len(inlier)-1, 3)
    #         while (len(set(random_index)) < 3):
    #             random_index = np.random.random_integers(0, len(inlier) - 1, 3)
    #             if (len(matches) < 3):
    #                 break
    #         a = np.zeros((6, 6), dtype=np.float64)
    #         b = np.zeros((6, 1), dtype=np.float64)
    #         for j in range(3):
    #             point1 = features1[matches[inlier[random_index[j]]][0]]
    #             point2 = features2[matches[inlier[random_index[j]]][1]]
    #             a[2 * j] = [point1[1], point1[0], 1, 0, 0, 0]
    #             a[2 * j + 1] = [0, 0, 0, point1[1], point1[0], 1]
    #             b[2 * j] = point2[1]
    #             b[2 * j + 1] = point2[0]
    #         last = np.array([0, 0, 1])
    #         t = np.dot(calculate_psedo_inverse(a), b)
    #         r = np.reshape(t, (2, 3))
    #         r = np.vstack((r, last))
    #         h_record[i] = r
    #
    #     h = np.mean(h_record, axis = 0)
    #     affine_xform = h


    # draw inlier_outlier image
    draw1 = draw_sbs_inlier_and_outlier(image1, image2, matches, features1,features2, inlier)

    cv2.imshow("ssift_match_features_inlier_outlier_img1_img2", draw1)
    # cv2.imwrite("ssift_matches_inlier outlier_wall1_img1_img2", draw1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # draw overlay image
    height, width = image2.shape[:2]

    warp_image = cv2.warpPerspective(image1, affine_xform, (width, height))
    draw2 = cv2.addWeighted(warp_image, 0.5, image2, 0.5, 1)
    cv2.imshow('ssift_overlay_img1_img2', draw2)
    # cv2.imwrite('ssift_overlay_wall1_wall3.png', draw2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return affine_xform