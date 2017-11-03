import cv2
from p1 import p1 as p1
from p2 import p2 as p2
from p3 import p3 as p3
from p4 import p4 as p4

import matplotlib.pyplot as plt
import numpy as np

# read 3 images
img1 = cv2.imread('many_objects_1.pgm',-1)
img2 = cv2.imread('two_objects.pgm', -1)
img3 = cv2.imread("many_objects_2.pgm", -1)

## apply p1 to 3 pgm images
p1_img1 = p1(img1, 128)
p1_img2 = p1(img2, 128)
p1_img3 = p1(img3, 128)
plt.imshow(p1_img1, cmap = "gray") # just showing one as demo
plt.title("p1_result_many_objects_1_demo")
plt.savefig("p1_result_many_objects_1_demo")
plt.show()



# apply p2 to 3 binary images (pre-processed by p1) get labled img
p2_img1 = p2(p1_img1) # many_objects_1.pgm   - labeled
p2_img2 = p2(p1_img2) # two_objcts.pgm       - labeled
p2_img3 = p2(p1_img3) # many_objects_2.pgm   - labeled
plt.imshow(p2_img1, cmap = "gray")
plt.title("p2_result_many_objects_1_demo")
plt.savefig("p2_result_many_objects_1_demo")
plt.show()


# apply p3 to img2 (two_objects.pgm)
database_out_2, p3_img2 = p3(p2_img2)
print database_out_2
plt.imshow(p3_img2, cmap = "gray")
plt.title("p3_result_two_objects_demo")
plt.savefig("p3_result_two_objecdts_demo")
plt.show()



# p4 - 1
p4_result = p4(p2_img1, database_out_2)
plt.imshow(p4_result, cmap = "gray")
plt.title("p4_result_many_objects_1")
plt.savefig("p4_result_1_many_objects_1")
plt.show()
# p4 - 2
p4_result_2 = p4(p2_img3, database_out_2)
plt.imshow(p4_result_2, cmap = "gray")
plt.title("p4_result_2_many_objects_2")
plt.savefig("p4_result_2_many_objects_2")
plt.show()

