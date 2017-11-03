import cv2
import matplotlib.pyplot as plt
from p5 import p5 as p5
from p6 import p6 as p6
from p7 import p7 as p7
from p8 import p8 as p8


# load images - single channel
hough_complex_1_single = cv2.imread('hough_complex_1.pgm', -1) # single - channel img
hough_simple_1_single = cv2.imread('hough_simple_1.pgm', -1) # single - channel img
hough_simple_2_single = cv2.imread('hough_simple_2.pgm', -1) # single - channel img




#######################################################################################################################
### (I) test for hough_complex_1.pgm, un-comment this block to run

edge_image = p5(hough_complex_1_single) # p5

edge_thresh_image, hough_image = p6(edge_image, 110) # p6

line_image = p7(hough_complex_1_single, hough_image, 110) # p7

p8_result = p8(hough_complex_1_single, hough_image, edge_thresh_image,110) # p8
#######################################################################################################################



#######################################################################################################################
### (II) test for hough_simple_1.pgm, un-comment this block to run

# edge_image = p5(hough_simple_1_single) # p5
# #
# edge_thresh_image, hough_image = p6(edge_image, 50) # p 6
#
# line_image = p7(hough_simple_1_single, hough_image, 110) # p7
#
# p8_result = p8(hough_simple_1_single, hough_image, edge_thresh_image,110) # p8
#######################################################################################################################



#######################################################################################################################
### (III) test for hough_simple_2.pgm, un-comment this block to run

# edge_image = p5(hough_simple_2_single) # p5
#
# edge_thresh_image, hough_image = p6(edge_image, 50) # p 6
# #
# line_image = p7(hough_simple_2_single, hough_image, 110) # p7
# # #
# p8_result = p8(hough_simple_2_single, hough_image, edge_thresh_image,110) # p8
#######################################################################################################################



#######################################################################################################################################
##### SHOW IMAGE ############### SHOW IMAGE ############### SHOW IMAGE ############### SHOW IMAGE ############### SHOW IMAGE ##########

# un_comment codes for above 3 blocks one by one to show result for each input image

# edge image
plt.imshow(edge_image, cmap = "gray")
plt.title("edge_image")
plt.show()

# edge_thresh_image
plt.imshow(edge_thresh_image, cmap = "gray")
plt.title("edge_thresh_image")
plt.show()

# hough_image
plt.imshow(hough_image, cmap = "gray")
plt.title("hough_image")
plt.show()

# line_image
plt.imshow(line_image, cmap = "gray")
plt.title("line_image")
plt.show()

# p8_result
plt.imshow(p8_result, cmap = "gray")
plt.title("p8_result")
plt.show()
##### SHOW IMAGE ############### SHOW IMAGE ############### SHOW IMAGE ############### SHOW IMAGE ############### SHOW IMAGE ##########
#######################################################################################################################################



