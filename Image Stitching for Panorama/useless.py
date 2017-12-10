import cv2
from detect_features import detect_features as detect_features
import numpy as np
import matplotlib.pyplot as plt
from detect_features import detect_features  as detect_features
import compute_affine_xform
import compute_proj_xform
import match_features




# loading images and convert to gray
b1_color = cv2.imread('bikes1.png')
b1 = cv2.cvtColor(b1_color, cv2.COLOR_BGR2GRAY)
b2_color = cv2.imread('bikes2.png')
b2 = cv2.cvtColor(b2_color, cv2.COLOR_BGR2GRAY)
b3_color = cv2.imread('bikes3.png')
b3 = cv2.cvtColor(b3_color, cv2.COLOR_BGR2GRAY)
g1_color = cv2.imread('graf1.png')
g1 = cv2.cvtColor(g1_color, cv2.COLOR_BGR2GRAY)
g2_color = cv2.imread('graf2.png')
g2 = cv2.cvtColor(g2_color, cv2.COLOR_BGR2GRAY)
g3_color = cv2.imread('graf3.png')
g3 = cv2.cvtColor(g3_color, cv2.COLOR_BGR2GRAY)
l1_color = cv2.imread('leuven1.png')
l1 = cv2.cvtColor(l1_color, cv2.COLOR_BGR2GRAY)
l2_color = cv2.imread('leuven2.png')
l2 = cv2.cvtColor(l2_color, cv2.COLOR_BGR2GRAY)
l3_color = cv2.imread('leuven3.png')
l3 = cv2.cvtColor(l3_color, cv2.COLOR_BGR2GRAY)
w1_color = cv2.imread('wall1.png')
w1 = cv2.cvtColor(w1_color, cv2.COLOR_BGR2GRAY)
w2_color = cv2.imread('wall2.png')
w2 = cv2.cvtColor(w2_color, cv2.COLOR_BGR2GRAY)
w3_color = cv2.imread('wall3.png')
w3 = cv2.cvtColor(w3_color, cv2.COLOR_BGR2GRAY)
#
#
# a = np.arange(36).reshape(6,6)
# b = np.arange(6).reshape(6,1)
# t = np.linalg.inv(a.T*a)*a.T*b
# a[5] = [0,0,0,0,0,0]
# min = 9999999
# record = t[0]
# print len(t)
# for i in range(len(t)):
#     det = (np.linalg.det(a.T*a*t[i] - b)) ** 2
#     if det < min:
#         min = det
#         record = t[i]
#         print min
# print "record is "
# print record
#
# print "t is "
# #
# print t
# a = np.array([[1,2,3,4,5,6],[2,3,4,5,6,7],[3,4,5,6,7,9],[4,5,7,5,4,3],[3,2,5,6,3,2],[0,0,0,0,0,0]])
# b = [1,3,5,3,2,3]
# test = np.linalg.inv(a.T*a) * a.T * b
# b = np.array([3,4,5])
# a = np.array([[1,2,3],[4,5,6],[0,0,0]])
#
# a = np.linalg.inv(np.dot())
# print test

#
# test = cv2.imread("bikes1.png")
# h = np.array([[1,1,1],[2,2,2],[0,0,1]])
# print len(test)
# print len(test[0])
# dst = cv2.warpPerspective(test,h,(1000, 700))
# load images
base = cv2.imread("wall3.png")
curr = cv2.imread("wall1.png")


# match, features1, features2
# feature detection
feature_coords1 = detect_features(w1)
feature_coords2 = detect_features(w3)
#
# # Match calculation
match = match_features.match_features(feature_coords1, feature_coords2, w1, w3)
# print "match is"
# print match
test = compute_affine_xform.compute_affine_xform(match, feature_coords1,feature_coords2,w1_color,w3_color)



transformation = compute_proj_xform.compute_proj_xform(match, feature_coords1, feature_coords2, w1_color, w3_color)

# transformation = compute_affine_xform.compute_affine_xform(match, feature_coords1, feature_coords2, g1_color, g2_color)








# find corresponding features in current photo
# curr_features = np.array([])RR
# curr_features, pyr_stati, _ = cv2.calcOpticalFlowPyrLK(base, curr, base_features, curr_features, flags=1)

# only add features for which a match was found to the pruned arrays
# base_features_pruned = []
# curr_features_pruned = []
# for index, status in enumerate(pyr_stati):
#     if status == 1:
#         base_features_pruned.append(base_features[index])
#         curr_features_pruned.append(curr_features[index])

# convert lists to numpy arrays so they can be passed to opencv function
# bf_final = np.asarray(base_features_pruned)
# cf_final = np.asarray(curr_features_pruned)

# find perspective transformation using the arrays of corresponding points
# transformation, hom_stati = cv2.findHomography(cf_final, bf_final, method=cv2.RANSAC, ransacReprojThreshold=1)

# transform the images and overlay them to see if they align properly
# not what I do in the actual program, just for use in the example code
# so that you can see how they align, if you decide to run it

height, width = base.shape[:2]

mod_photo = cv2.warpPerspective(curr, transformation, (width, height))
new_image = cv2.addWeighted(mod_photo, .5, base, .5, 1)

cv2.imshow("test", new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()