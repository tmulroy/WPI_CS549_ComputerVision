import numpy as np
import cv2 as cv


def detect_coins(image):
    # Convert to grayscale and blur to reduce noise
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    # Perform Hough Circle Transform
    rows = gray.shape[0]
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
        # print(radius)
        cv.circle(image, center, radius, (255, 0, 255), 3)

    return image, circles


def recognize_coins_ratios(src, circles):
    # Note: Using SIFT/SURF and Brute Force/Flann doesn't result in good matches.
    # I use the ratio of the coin diameters
    # print(f'circles info: {circles}')
    coins = {
        'penny': {
            'diameter': 19.05,
            'value': 0.01,
            'ratio': 1.0636,
            'count': 0
        },
        'nickel': {
            'diameter': 21.21,
            'value': 0.05,
            'ratio': 1.1842,
            'count': 0
        },
        'dime': {
            'diameter': 19.05,
            'value': 0.1,
            'ratio': 1,
            'count': 0
        },
        'quarter': {
            'diameter': 26.5,
            'value': 0.25,
            'ratio': 1.4796,
            'count': 0
        }
    }

    ratios = np.asarray([1.0636, 1.1842, 1, 1.4796])
    ratios_names = ['penny', 'nickel', 'dime', 'quarter']
    diameters = []
    sum = 0
    for circle in circles[0]:
        diameter = circle[2]*2
        diameters.append(diameter)

    min_diameter = min(diameters)

    for coin in circles[0]:
        print('\n')
        diameter = coin[2]*2
        ratio = round(diameter/min_diameter, 4)
        idx = (np.abs(ratios - ratio)).argmin()
        coins[ratios_names[idx]]['count'] += 1
        sum += coins[ratios_names[idx]]['value']
    return round(sum, 2)


# Part 2
def hdr():
    img_fn = ["IMAGE_1.JPG", "IMAGE_2.JPG", "IMAGE_3.JPG"]
    img_list = [cv.imread(fn) for fn in img_fn]
    merge_mertens = cv.createMergeMertens()
    res_mertens = merge_mertens.process(img_list)
    res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
    cv.imwrite("fusion_mertens.jpg", res_mertens_8bit)



if __name__ == '__main__':
    filename = 'coins_test.png'
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    circle_img, circles_info = detect_coins(src)
    sum = recognize_coins_ratios(src, circles_info)
    print(f'${sum}')
    hdr()

