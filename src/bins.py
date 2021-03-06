import cv2
import numpy as np
import copy
import sys
import time

###################################################
colorBalanceRatio = 5
lb = []
lc = []
le = []
ld = True
lf = []
lg = []
################################
# helpers and driver
################################


def show(img, msg="image", ana=True):
    cv2.imshow(msg, img)
    if ana:
        analysis(img)
    cv2.waitKey(0)


def show2(img, msg="image2", ana=True):

    cv2.imshow(msg, img/255)
    if ana:
        analysis(img)
    cv2.waitKey(100)


def open(name, path1):
    #"/Users/rongk/Downloads/test.jpg"):
    if name == "d":
        path0 = "/home/dhyang/Desktop/Vision/vision/Images/buoyComp/"
    #path = "/Users/rongk/Downloads/Vision-master/Vision-master/RoboticsImages/images/training15.png"
    #path = "/Users/rongk/Downloads/Vision-master/Vision-master/RoboticsImages/03.jpg"
    else:
        path0 = "/Users/rongk/Downloads/visionCode/Vision/bins/"
    path2 = ".jpg"
    path = path0+path1+path2
    img = cv2.imread(path)
    print(path)
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    return img


def analysis(img):
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])
    for i, col in enumerate(("b", "g", "r")):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()
######################################
# main program removebackscatter
#######################################


def reflect(image, blkSize=10*10, patchSize=8, lamb=10, gamma=1.7, r=10, eps=1e-6, level=5):
    image = np.array(image, np.float32)
    bgr = cv2.split(image)
    #show(bgr[2]/255,"initial red",False)
    # image decomposition, probably key
    RL = IDilluRefDecompose(image)
    RL = FsimpleColorBalance(RL, colorBalanceRatio)  # checked
    # show2(RL,"color corrected reflective") #checked
    bgr = cv2.split(RL)
    #show(bgr[0]/255,"RL blue",False)
    #show(bgr[1]/255,"RL green",False)
    #show(bgr[2]/255,"RL red",False)
    return RL
####################################################
# Img Decompose: weighted image decompose
####################################################


def IDilluRefDecompose(img):
    RList = []
    bgr = cv2.split(img)
    for cnl in bgr:
        rlcnl = copy.deepcopy(cnl)
        maxVal = np.asmatrix(cnl).max()
        k = np.multiply(cnl, .5/maxVal)
        rlcnl = np.multiply(k, rlcnl)
        RList.append(rlcnl)
    Rl = cv2.merge(RList)
    return Rl
######################################
# Filter
######################################


def FsimpleColorBalance(img, percent):
    start_time = time.time()
    if percent <= 0:
        percent = 5
    img = np.array(img, np.float32)
    rows = img.shape[0]
    cols = img.shape[1]
    chnls = img.shape[2]
    halfPercent = percent/200
    if chnls == 3:
        channels = cv2.split(img)
    else:
        channels = copy.deepcopy(img)
        # Not sure
    channels = np.array(channels)

    for i in range(chnls):
        # find the low and high precentile values based on input percentile
        flat = np.array(channels[i].flat)
        flat.sort()
        lowVal = flat[int(np.floor(len(flat)*halfPercent))]

        topVal = flat[int(np.ceil(len(flat)*(1-halfPercent)))]
        channels[i] = np.where(channels[i] > lowVal, channels[i], lowVal)
        channels[i] = np.where(channels[i] < topVal, channels[i], topVal)
        channels[i] = cv2.normalize(
            channels[i], channels[i], 0.0, 255.0/2, cv2.NORM_MINMAX)
        channels[i] = np.float32(channels[i])

    result = cv2.merge(channels)
    return result
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def binarization(gray):
    ret, thresh1 = cv2.threshold(gray,200, 255, cv2.THRESH_BINARY)
    thresh1 = cv2.bitwise_not(thresh1)
    return thresh1


def getLines(newImg):
    csums = np.sum(newImg, axis=0)
    csums1 = copy.deepcopy(csums)
    lineLocs = []
    leeway = 100
    for i in range(2):
        lineLocs.append([np.argmin(csums), csums[np.argmin(csums)]])
        lhs = lineLocs[i][0]-leeway
        rhs = lineLocs[i][0]+leeway
        if lhs < 0:
            lhs = 0
        if rhs >= newImg.shape[1]:
            rhs = newImg.shape[1]-1
        csums[lhs:rhs] = 1000000
    if True:
        plt.plot(csums1)
        for i in range(len(lineLocs)):
            plt.axvline(x=lineLocs[i][0], color='r', linewidth=1)
        plt.show()
    newImg = cv2.cvtColor(newImg, cv2.COLOR_GRAY2BGR)
    #error = lineLocs[2][1]-(lineLocs[0][1]+lineLocs[1][1])/2
    error = 0
    return lineLocs, error


def plotLines(lineLocs, original):
    for i in range(2):
        cv2.line(original, (lineLocs[i][0], 0),
                 (lineLocs[i][0], original.shape[0]), (0, 255, 0), 3)
    norm = 0
    center = 0
    for k in range(len(lineLocs)):
        center = center + (50000-lineLocs[k][1])*lineLocs[k][0]
        norm = norm + (50000-lineLocs[k][1])
    #center = (int) (center/norm)
    center = (int)((lineLocs[0][0]+lineLocs[1][0])/2)
    cv2.line(original, (center, 0),
             (center, original.shape[0]), (0, 0, 255), 1)
    return original


def segment(image):
    mdpt = (int)(image.shape[0]/2)
    striph = 150
    return image[mdpt - striph: mdpt + striph, :]


def adjust(image):
    alphah = 5
    alphas = 5
    alphav = 5

    h, s, v = cv2.split(image)
    new_image = np.zeros(image.shape, image.dtype)
    h1, s1, v1 = cv2.split(new_image)

    maximum = h.mean()
    #maximum = h.min()
    beta = 127-alphah*maximum  # Simple brightness control
    h1 = cv2.convertScaleAbs(h, alpha=alphah, beta=beta)

    maximum = s.mean()
    beta = 127-alphas*maximum  # Simple brightness control
    s1 = cv2.convertScaleAbs(s, alpha=alphas, beta=beta)

    maximum = v.mean()
    beta = 127-alphav*maximum  # Simple brightness control
    v1 = cv2.convertScaleAbs(v, alpha=alphav, beta=beta)

    new_image = cv2.merge([h1, s1, v1])
    return new_image

def adjustYUV(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    alphah = 3
    alphas = 0
    alphav = 3

    h, s, v = cv2.split(image)
    new_image = np.zeros(image.shape, image.dtype)
    h1, s1, v1 = cv2.split(new_image)

    maximum = h.mean()
    #maximum = h.min()
    beta = -alphah*maximum  # Simple brightness control
    h1 = cv2.convertScaleAbs(h, alpha=alphah, beta=beta)

    maximum = s.mean()
    beta = 127-alphas*maximum  # Simple brightness control
    s1 = cv2.convertScaleAbs(s, alpha=alphas, beta=beta)

    maximum = v.mean()
    beta = 127-alphav*maximum  # Simple brightness control
    v1 = cv2.convertScaleAbs(v, alpha=alphav, beta=beta)

    new_image = cv2.merge([h1, s1, v1])
    new_image = cv2.cvtColor(new_image, cv2.COLOR_YUV2BGR)
    return new_image

def boundingRectangle(original,thresh):
    contours,h = cv2.findContours(thresh,1,2)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(original,[box],0,(0,0,255))

def fill(original,thresh):
    contours,h = cv2.findContours(thresh,1,2)
    img = np.ones([original.shape[0],original.shape[1],3], dtype=np.uint8)*255
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(cnt)
        if area > 100 and area < 10000:
            cv2.drawContours(img,[box],0,(0,0,0),thickness=cv2.FILLED)
    return img
############################################


def getMask(img):
    lower_green = np.array([0,0,0])
    upper_green = np.array([255,180,255])
    mask = cv2.inRange(img, lower_green, upper_green)
    mask = cv2.bitwise_not(mask)
    return mask

def floodfill(img):
    im_floodfill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = img | im_floodfill_inv
    return im_out


def mainImg(img):
    start_time = time.time()
    original = img
    origin = copy.deepcopy(original)

    o1 = original

    #cv2.imshow("original", origin)

    original = reflect(original)
    show2(original, "filtered", False)
    segmented = adjustYUV(original)
    segmented = adjust(segmented)

    #get mask
    mask = getMask(segmented)

    #binarization
    newImg1 = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)

    newImg1 = binarization(newImg1)
    newImg1 = cv2.bitwise_not(mask)
    #newImg1 = floodfill(newImg1)
    newImg1 = fill(o1,newImg1)
    newImg1 = cv2.cvtColor(newImg1, cv2.COLOR_BGR2GRAY)
    #boundingRectangle(o1,newImg1)
    segmented = cv2.cvtColor(segmented, cv2.COLOR_HSV2RGB)
    cv2.imshow("alpha", segmented)
    cv2.imshow("binarization", newImg1)
    #cv2.imshow("background subtraction", redSpace)
    end_time = time.time()
    cv2.imshow("result", o1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return segmented
####################################################
#########################################################
#####################################################


def main():
    img = open(sys.argv[1], sys.argv[2])
    mainImg(img)
    print("Vision Code")


if __name__ == "__main__":
    main()

