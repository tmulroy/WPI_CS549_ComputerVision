import cv2 as cv

# Instantiation
sift = cv.SIFT_create()
surf = cv.xfeatures2d.SURF_create(5000)
bf = cv.BFMatcher()

# FLANN Parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)

# Images
book = cv.imread('book.jpg')
table = cv.imread('table.jpg')
gray_book = cv.cvtColor(book, cv.COLOR_BGR2GRAY)
gray_table = cv.cvtColor(table, cv.COLOR_BGR2GRAY)

# SIFT Feature Detection and Description
kp_sift_book, des_sift_book = sift.detectAndCompute(gray_book, None)
kp_sift_table, des_sift_table = sift.detectAndCompute(gray_table, None)
img_sift_book = cv.drawKeypoints(gray_book, kp_sift_book, book, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_sift_table = cv.drawKeypoints(gray_table, kp_sift_table, table, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# SURF Feature Detection and Description
kp_surf_book, des_surf_book = surf.detectAndCompute(gray_book, None)
kp_surf_table, des_surf_table = surf.detectAndCompute(gray_table, None)
img_surf_book = cv.drawKeypoints(gray_book, kp_surf_book, book, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_surf_table = cv.drawKeypoints(gray_table, kp_surf_table, table, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# Brute Force Feature Matching
# SIFT
bf_sift_matches = bf.match(des_sift_book, des_sift_table)
bf_sift_matches = sorted(bf_sift_matches, key=lambda x:x.distance)
sift_bf = cv.drawMatches(gray_book, kp_sift_book, gray_table, kp_sift_table, bf_sift_matches[:10], None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# SURF
bf_surf_matches = bf.match(des_surf_book, des_surf_table)
bf_surf_matches = sorted(bf_surf_matches, key=lambda x:x.distance)
surf_bf = cv.drawMatches(gray_book, kp_surf_book, gray_table, kp_surf_table, bf_surf_matches[:10], None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


# FLANN Feature Matching
# SIFT
flann_sift_matches = flann.knnMatch(des_sift_book, des_sift_table, k=2)
matchesMask_sift = [[0,0] for i in range(len(flann_sift_matches))]
# ratio test
for i,(m,n) in enumerate(flann_sift_matches):
    if m.distance < 0.7*n.distance:
        matchesMask_sift[i]=[1,0]

draw_params_flann_sift = dict(matchColor = (0,255,0),
                               singlePointColor = (255,0,0),
                               matchesMask = matchesMask_sift,
                               flags = cv.DrawMatchesFlags_DEFAULT)
flann_sift_img = cv.drawMatchesKnn(gray_book, kp_sift_book, gray_table, kp_sift_table, flann_sift_matches,None, **draw_params_flann_sift)

# SURF
flann_surf_matches = flann.knnMatch(des_surf_book, des_surf_table, k=2)
matchesMask_surf = [[0,0] for i in range(len(flann_surf_matches))]
# ratio test
for i,(m,n) in enumerate(flann_surf_matches):
    if m.distance < 0.7*n.distance:
        matchesMask_surf[i]=[1,0]
draw_params_flann_surf = dict(matchColor = (0,255,0),
                               singlePointColor = (255,0,0),
                               matchesMask = matchesMask_surf,
                               flags = cv.DrawMatchesFlags_DEFAULT)
flann_surf_img = cv.drawMatchesKnn(gray_book, kp_surf_book, gray_table, kp_surf_table, flann_surf_matches,None, **draw_params_flann_surf)


# Show Images
# SIFT Keypoints
cv.imshow('SIFT Book Keypoints', img_sift_book)
cv.imshow('SIFT Table Keypoints', img_sift_table)

# SURF Keypoints
cv.imshow('SURF Book Keypoints', img_surf_book)
cv.imshow('SURF Table Keypoints', img_surf_table)

# SIFT
# Brute Force
cv.imshow('SIFT with Brute Force', sift_bf)

# FLANN
cv.imshow('SIFT with FLANN', flann_sift_img)


# SURF
# Brute Force
cv.imshow('SURF with Brute Force', surf_bf)

# FLANN
cv.imshow('SURF with FLANN', flann_surf_img)
cv.waitKey()