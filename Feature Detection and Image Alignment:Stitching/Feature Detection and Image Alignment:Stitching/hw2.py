import cv2
from detect_features import detect_features as detect_features
import match_features
import compute_affine_xform
import ssift_descriptor
import compute_proj_xform
import match_features_ratio



# loading images
b1= cv2.imread('bikes1.png')
b2= cv2.imread('bikes2.png')
b3 = cv2.imread('bikes3.png')
g1 = cv2.imread('graf1.png')
g2 = cv2.imread('graf2.png')
g3 = cv2.imread('graf3.png')
l1 = cv2.imread('leuven1.png')
l2 = cv2.imread('leuven2.png')
l3 = cv2.imread('leuven3.png')
w1 = cv2.imread('wall1.png')
w2 = cv2.imread('wall2.png')
w3 = cv2.imread('wall3.png')
#

################################################################################################################################
# Step 1: chose two images here
img1 = b1
img2 = b2


################################################################################################################################
# Step 2: feature detection for two images
feature_coords1 = detect_features(img1)
feature_coords2 = detect_features(img2)


################################################################################################################################
# Step 3: Match calculation using NCC with mutual marriage, image saved as match_features_{img1_name}_{img2_name}.png
match = match_features.match_features(feature_coords1, feature_coords2, img1, img2)


################################################################################################################################
# Step 4 - 1: compute affine transformation, 2 images displayed: (1) "affine__inlier_outlier_{img1_name}_{img2_name}.png" shows inlier
# and outlier (red line: outlier, green line: inlier) and (2) "Affine_Overlay_{img1_name}_{img2_name}.png", overlay 2 image to see how
# they fit with each other
compute_affine_xform.compute_affine_xform(match, feature_coords1, feature_coords2, img1, img2)


################################################################################################################################
# Step 4-2: compute proj transformation. Similarly, 2 images displayed: (1) "proj_matches_inlier_outlier_{img1_name}_{img2_name}.png" shows inlier
# and outlier (red line: outlier, green line: inlier) and (2) "Proj_Overlay_{img1_name}_{img2_name}.png", overlay 2 image to see how
# they fit with each other
compute_proj_xform.compute_proj_xform(match, feature_coords1, feature_coords2, img1, img2)


# step 5: compute ssift descriptors,  and match 2 images using match_features_ratio.py, and  displayed (1) "ssift_match_inlier_outlier_{img1_name}_{img2_name}.png"
# and (2) "sswift_overlay_{img1_name}_{img2_name}.png"

descriptors1 = ssift_descriptor.ssift_descriptor(feature_coords1, img1)
descriptors2 = ssift_descriptor.ssift_descriptor(feature_coords2, img2)
ssift_match = match_features_ratio.match_features_ratio(feature_coords1,descriptors1,feature_coords2,descriptors2)
print ssift_match

# 2 images: (1) ssift match features (with ratio matching) with inlier and outlier (2) ssift overlay image
compute_affine_xform.compute_affine_xform_for_ssift(ssift_match,feature_coords1,feature_coords2,img1, img2)

