import cv2 as cv
from matplotlib import pyplot as plt

def one():
    img  = cv.imread('test.jpg')
    cv.imshow('image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def two():
    img = cv.imread('test.jpg')
    img[:,:,0] = 0
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
def three():
    img = cv.imread('test.jpg',cv.IMREAD_GRAYSCALE)

    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
def four():
    img = cv.imread('test.jpg')
    blur = cv.GaussianBlur(img, (5, 5), 0)
    cv.imshow('image', blur)
    cv.waitKey(0)
    cv.destroyAllWindows()
def five():
    img = cv.imread('test.jpg')
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    angle90 = 90
    scale = 1.0
    M = cv.getRotationMatrix2D(center, angle90, scale)
    rotated90 = cv.warpAffine(img, M, (h, w))
    cv.imshow('image', rotated90)
    cv.waitKey(0)
    cv.destroyAllWindows()
def six():
    img = cv.imread('test.jpg')
    small = cv.resize(img, (0, 0), fx=0.5, fy=1.0)
    cv.imshow('image', small)
    cv.waitKey(0)
    cv.destroyAllWindows()

def seven():
    img = cv.imread('test.jpg', 0)
    edges = cv.Canny(img, 100, 200)
    cv.imshow('image',edges)
    cv.waitKey(0)
    cv.destroyAllWindows()
def eight():
    pass