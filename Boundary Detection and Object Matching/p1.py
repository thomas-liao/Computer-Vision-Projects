import numpy as np

def p1(gray_in, thresh_val): # return binary_out
    height = len(gray_in)
    width = len(gray_in[0])
    ret_img = np.zeros((height, width), dtype = np.uint64) # must set it up higher otherwise bug in labeling process
    for i in range(height):
        for j in range(width):
            if(gray_in[i][j] <= thresh_val):
                ret_img[i][j] = 0
            else:
                ret_img[i][j] = 255
    return ret_img





