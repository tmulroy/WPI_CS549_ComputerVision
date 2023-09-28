import cv2 as cv

# Instantiation
sift = cv.SIFT_create()
surf = cv.xfeatures2d.SURF_create(25000)
bf = cv.BFMatcher()


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
# SURF

# Show Images
# SIFT Keypoints
cv.imshow('SIFT Book Keypoints', img_sift_book)
cv.imshow('SIFT Table Keypoints', img_sift_table)

# SURF Keypoints
cv.imshow('SURF Book Keypoints', img_surf_book)
cv.imshow('SURF Table Keypoints', img_surf_table)

# SIFT with Brute Force Matching
# cv.imshow('SIFT with Brute Force', bf_sift_matches)
cv.imshow('SIFT with Brute Force', sift_bf)

# SIFT with FLANN Matching

# SURF with Brute Force Matching
# cv.imshow('SURF with Brute Force', bf_surf_matches)
# SURF with FLANN Matching

cv.waitKey()