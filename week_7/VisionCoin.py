import numpy as np
import cv2 as cv


def detect_coins(image):
    # Convert to grayscale and blur to reduce noise
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    # Perform Hough Circle Transform
    rows = gray.shape[0]
    print(rows)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows/4,
                              param1=100, param2=50,
                              minRadius=100, maxRadius=200)

    if circles is not None:
        circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv.circle(image, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        diameter = radius*2
        print(diameter)
        # print(radius)
        cv.circle(image, center, radius, (255, 0, 255), 3)

    return image, circles

def recognize_coins(src, info):
    # Uses known images of coins to determine if they are in the input image
    penny_img = cv.imread('penny.png')
    nickel_img = cv.imread('nickel.png')
    dime_img = cv.imread('dime.png')
    quarter_img = cv.imread('quarter.png')

    # Convert Known Coin Images to Gray and Blur to Reduce Noise
    gray_penny = cv.cvtColor(penny_img, cv.COLOR_BGR2GRAY)
    gray_penny = cv.medianBlur(gray_penny, 5)

    gray_nickel = cv.cvtColor(nickel_img, cv.COLOR_BGR2GRAY)
    gray_nickel = cv.medianBlur(gray_nickel, 5)

    gray_dime = cv.cvtColor(dime_img, cv.COLOR_BGR2GRAY)
    gray_dime = cv.medianBlur(gray_dime, 5)

    gray_quarter = cv.cvtColor(quarter_img, cv.COLOR_BGR2GRAY)
    gray_quarter = cv.medianBlur(gray_quarter, 5)

    gray_input = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray_input = cv.medianBlur(gray_input, 5)

    # Instantiate Feature Detection and Matching Algorithm Classes
    sift = cv.SIFT_create()
    bf = cv.BFMatcher()
    surf = cv.xfeatures2d.SURF_create(1000)

    # SIFT Feature Detection and Description
    kp_penny, des_penny = surf.detectAndCompute(gray_penny, None)
    kp_nickel, des_nickel = surf.detectAndCompute(gray_nickel, None)
    kp_dime, des_dime = surf.detectAndCompute(gray_dime, None)
    kp_quarter, des_quarter = surf.detectAndCompute(gray_quarter, None)
    kp_input, des_input = surf.detectAndCompute(gray_input, None)

    # Brute Force Matching
    # Penny
    penny_matches = bf.match(des_penny, des_input)
    penny_matches = sorted(penny_matches, key=lambda x:x.distance)
    penny_matches_img = cv.drawMatches(gray_penny, kp_penny, src, kp_input, penny_matches[:10], None,
                             flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('penny matches', penny_matches_img)
    cv.waitKey()

    # Nickel
    nickel_matches = bf.match(des_nickel, des_input)
    nickel_matches = sorted(nickel_matches, key=lambda x: x.distance)
    nickel_matches_img = cv.drawMatches(gray_nickel, kp_nickel, src, kp_input, nickel_matches[:10], None,
                                       flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('nickels', nickel_matches_img)
    cv.waitKey()

    # Note: Using SIFT/SURF and Brute Force/Flann doesn't result in good matches.
    # I use the ratio of the coin diameters



if __name__ == '__main__':
    filename = 'coins_test.png'
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    circle_img, circles_info = detect_coins(src)
    recognize_coins(src, circles_info)
    # cv.imshow('circles', circle_img)
    # cv.waitKey()

