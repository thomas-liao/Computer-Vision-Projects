import cv2
import numpy as np
import matplotlib.pyplot as plt


def ssift_descriptor(feature_coords, image):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        feature_coords (list of tuples): list of (row,col) tuple feature coordinates from image
        image (numpy.ndarray): The input image to compute ssift descriptors on. Note: this is NOT the image name or image path.
    Returns:
        descriptors (dictionary{(row,col): 128 dimensional list}): the keys are the feature coordinates (row,col) tuple and
                                                                   the values are the 128 dimensional ssift feature descriptors.
    """
    # data structure: descriptor[point[0], point[1]] =  128-dimension feature
    descriptors = dict()
    # convert to gray scale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # normalize vector to unit 1
    def normalize(vec):
        sum = 0
        for i in range(len(vec)):
            sum += vec[i] ** 2
        dev = np.sqrt(sum)
        vec /= dev
        return vec

    # return 4*4*8 feature for any given point in feature_coords
    shape = 20
    height = len(image)
    width = len(image[0])
    image = image.astype(np.float64)
    # smooth the image with 3 by 3 gausian filter
    gausian_filter = cv2.getGaussianKernel(3,1)
    image = cv2.filter2D(image, -1, gausian_filter)
    image = cv2.filter2D(image, -1, gausian_filter.T)

    # calculate gradient:
    sobel_33_x = np.array([[-1, 0, 1], [-2, 0, 2.], [-1., 0., 1.]], dtype=np.float64)
    sobel_33_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    ix = cv2.filter2D(image, -1, sobel_33_x)
    iy = cv2.filter2D(image, -1, sobel_33_y)
    gradient = np.sqrt(ix**2 + iy**2)
    orientation = np.arctan2(ix, iy) + 2* np.pi # convert to positive

    for itr in range(len(feature_coords)):
        point = feature_coords[itr]
        descriptors[point[1], point[0]] = []
        # check if the point is near boundary
        if point[0] < shape or point[0] > image.shape[0] - shape or point[1] < shape or point[1] > image.shape[
            1] - shape or point[0] < shape or point[0] > image.shape[0] - shape or point[1] < shape or point[1] > \
                        image.shape[1] - shape:
            descriptors[point[1], point[0]] = []
            continue
        # get matrix centered at the point
        matrix_gradient = gradient[point[0] - shape:point[0]+shape, point[1]-shape:point[1] + shape]
        matrix_orientation = orientation[point[0] - shape:point[0]+shape, point[1]-shape:point[1] + shape]
        # gausian filter to blur matrix_gradient - more weight near center point
        gausian_filter2 = cv2.getGaussianKernel(40,1)
        matrix_gradient = cv2.filter2D(matrix_gradient, -1, gausian_filter2)
        matrix_gradient = cv2.filter2D(matrix_gradient, -1, gausian_filter2.T)


        temp1 = [] # 128-d vector
        for i in range(4):
            for j in range(4):
                # calculate the 8 - d feature
                temp2 = [0]*8
                for k in range(i*10 + 10):
                    for l in range(j * 10 + 10):
                        vote_orientation = int(round(matrix_orientation[k][l] * 4 / np.pi) %8)
                        print vote_orientation
                        temp2[vote_orientation] += matrix_gradient[k][l]
                temp1.extend(temp2)
        #normalization - threshold - normalization for temp1
        temp1 = normalize(temp1)
        for i in range(len(temp1)):
            if temp1[i] > 0.2:
                temp1[i] = 0.2
        temp1 = normalize(temp1)
        descriptors[point[1], point[0]] = temp1
    return descriptors







