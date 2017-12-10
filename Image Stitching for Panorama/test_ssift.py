import cv2
from detect_features import detect_features as detect_features
import match_features
import compute_affine_xform
import re_write
import ssift_descriptor_compare
import ssift_descriptor
import compute_proj_xform
import match_features_ratio
import numpy as np
import compute_proj_xform
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

# feature detection
feature_coords1 = detect_features(g1)
feature_coords2 = detect_features(g2)
print "f1"
print feature_coords1
print "f2"
print feature_coords2
descriptors1 = ssift_descriptor.ssift_descriptor(feature_coords1, g1)
descriptors2 = ssift_descriptor.ssift_descriptor(feature_coords2, g2)



match = match_features_ratio.match_features(feature_coords1,descriptors1,feature_coords2,descriptors2, g1_color, g2_color)





print "descriptors1 is "
print descriptors1
print "This is match"
print match

compute_proj_xform.compute_proj_xform(match, feature_coords1, feature_coords2, g1_color,g2_color)
# cv2.imshow("image", test)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

base = cv2.imread("graf2.png")
curr = cv2.imread("graf1.png")

# match, features1, features2
# feature detection
feature_coords1 = detect_features(g1)
feature_coords2 = detect_features(g2)
#
# # Match calculation
# match = match_features.match_features(feature_coords1, feature_coords2, g1, g2)
# print "match is"
# print match
# test = compute_affine_xform.compute_affine_xform(match, feature_coords1,feature_coords2,b1_color,b1_color)

transformation = compute_proj_xform.compute_proj_xform(match, feature_coords1, feature_coords2, g1_color, g2_color)

# transformation = compute_affine_xform.compute_affine_xform(match, feature_coords1, feature_coords2, g1_color, g2_color)

# convert to grayscale
base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

# find the coordinates of good features to track  in base
base_features = cv2.goodFeaturesToTrack(base_gray, 3000, .01, 10)

# find corresponding features in current photo
curr_features = np.array([])
curr_features, pyr_stati, _ = cv2.calcOpticalFlowPyrLK(base, curr, base_features, curr_features, flags=1)

# only add features for which a match was found to the pruned arrays
base_features_pruned = []
curr_features_pruned = []
for index, status in enumerate(pyr_stati):
    if status == 1:
        base_features_pruned.append(base_features[index])
        curr_features_pruned.append(curr_features[index])

# convert lists to numpy arrays so they can be passed to opencv function
bf_final = np.asarray(base_features_pruned)
cf_final = np.asarray(curr_features_pruned)

# find perspective transformation using the arrays of corresponding points
# transformation, hom_stati = cv2.findHomography(cf_final, bf_final, method=cv2.RANSAC, ransacReprojThreshold=1)

# transform the images and overlay them to see if they align properly
# not what I do in the actual program, just for use in the example code
# so that you can see how they align, if you decide to run it

height, width = curr.shape[:2]

mod_photo = cv2.warpPerspective(curr, transformation, (width, height))
new_image = cv2.addWeighted(mod_photo, .5, base, .5, 1)

cv2.imshow("test", new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

