import cv2
from matplotlib import pyplot as plt
import numpy as np


def one():
    img = cv2.imread('test.jpg')
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def two():
    img = cv2.imread('test.jpg')
    img[:, :, 0] = 0
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def three():
    img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def four():
    img = cv2.imread('test.jpg')
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imshow('image', blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def five():
    img = cv2.imread('test.jpg')
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    angle90 = 90
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle90, scale)
    rotated90 = cv2.warpAffine(img, M, (h, w))
    cv2.imshow('image', rotated90)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def six():
    img = cv2.imread('test.jpg')
    small = cv2.resize(img, (0, 0), fx=0.5, fy=1.0)
    cv2.imshow('image', small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def seven():
    img = cv2.imread('test.jpg', 0)
    edges = cv2.Canny(img, 100, 200)
    cv2.imshow('image', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def eight():
    img = cv2.imread('test.jpg')
    gray = cv2.cv2tColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def nine():
    imagePath = 'test.jpg'
    cascPath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Faces found", image)
    cv2.waitKey(0)


def ten():
    cap = cv2.VideoCapture('test.avi')
    cnt = 0
    while (cap.isOpened()):
        cnt += 1
        ret, frame = cap.read()
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (100, 100)
        fontScale = 2
        fontColor = (255, 255, 255)
        lineType = 2

        cv2.putText(frame, 'index :' + str(cnt),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imshow('frame ',frame)
        cv2.waitKey(500)


ten()