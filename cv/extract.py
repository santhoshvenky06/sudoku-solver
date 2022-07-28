import numpy 
import cv2
import operator
from matplotlib import pyplot as plt

#processing the raw image to obtain only the sudoku puzzle
#invert the image colours
#and finally dilate
def processPuzzle(image, dilate=true):
    process = cv2.GaussianBlur(image.copy(), (9,9), 0)
    process = cv2.adaptiveThreshold(process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    process = cv2.bitwise_not(process,process)

    if dilate: 
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
        process = cv2.dilate(process, kernel)
    
    return process

#find all squares in the image and sort them in descending order of area and choose the largest one
def findCorners(image):
    contour, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = sorted(contour, key=cv2.contourArea, reverse=True)
    largestSquare = contour[0]

    br, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in largestSquare]), key=operator.itemgetter(1))
    tl, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in largestSquare]), key=operator.itemgetter(1))
    bl, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in largestSquare]), key=operator.itemgetter(1))
    tr, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in largestSquare]), key=operator.itemgetter(1))

    return [largestSquare[tl][0], largestSquare[tr][0], largestSquare[br][0], largestSquare[bl][0]]