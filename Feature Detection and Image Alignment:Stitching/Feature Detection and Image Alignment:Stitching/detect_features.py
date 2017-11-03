import cv2
import numpy as np
import matplotlib.pyplot as plt
from nonmaxsuppts import nonmaxsuppts as nonmaxsuppts

def detect_features(image):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        image (numpy.ndarray): The input image to detect features on. Note: this is NOT the image name or image path.
    Returns:
        pixel_coords (list of tuples): A list of (row,col) tuples of detected feature locations in the image
    """
    pixel_coords = list()
    # convert image to gray scale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # we first smooth the image a little bit by 3*3 gausian kernel with sigma = 1
    # because gausian kernal is linear separable, which is, we are filter2D 2 times with gausian_kernel and np.transpose(gausian_kernel)
    smooth_kernel_col = cv2.getGaussianKernel(3,1)
    smooth_kernel_row = np.transpose(smooth_kernel_col)
    # now we can smooth image
    img = image.astype(np.float64)
    img = cv2.filter2D(img, -1, smooth_kernel_col)
    img = cv2.filter2D(img, -1,smooth_kernel_row)
    # now get gradient for x and y
    # sobel_33_x = np.array([[-1, 0, 1], [-2, 0, 2.], [-1., 0., 1.]], dtype=np.float64)
    # sobel_33_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    sobel_55_x = np.array([[-1, -2, 0, 2, 1], [-2, -3, 0, 3, 2], [-3, -5, 0, 5, 3], [-2, -3, 0, 3, 2], [-1, 2, 0, 2, 1]])
    sobel_55_y = np.array([[1, 2, 3, 2, 1], [2, 3, 5, 3, 2], [0, 0, 0, 0, 0], [-2, -3, -5, -3, -2], [-1, -2, -3, -2, -1]])
    ix = cv2.filter2D(img, -1, sobel_55_x)
    iy = cv2.filter2D(img, -1, sobel_55_y)
    # w(x, y)
    gausian_kernel_col = cv2.getGaussianKernel(15, 1)
    gausian_kernel_row = np.transpose(gausian_kernel_col)
    ix2 = np.multiply(ix, ix)
    iy2 = np.multiply(iy, iy)
    ixiy = np.multiply(ix, iy)

    # convolution with gausian filter
    ix2 = cv2.filter2D(ix2, -1, gausian_kernel_col)
    ix2 = cv2.filter2D(ix2, -1, gausian_kernel_row)
    iy2 = cv2.filter2D(iy2, -1, gausian_kernel_col)
    iy2 = cv2.filter2D(iy2, -1, gausian_kernel_row)
    ixiy = cv2.filter2D(ixiy, -1, gausian_kernel_col)
    ixiy = cv2.filter2D(ixiy, -1, gausian_kernel_row)

    height, width = img.shape
    cs = np.zeros(img.shape, np.float64)
    k = 0.05  # from 0.04 to 0.06

    # det M = |Ixx  Ixy ; Ixy  iyy|
    for i in range(height):
        for j in range(width):
            detM = ix2[i][j] * iy2[i][j] - ixiy[i][j] ** 2
            traceM = ix2[i][j] + iy2[i][j]
            cs[i][j] = detM - k * traceM ** 2

    a = int(cs.max() / 10)
    pixel_coords = nonmaxsuppts(cs, 7, a * 0.5)
    return pixel_coords











