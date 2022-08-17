from ctypes import resize
from random import seed
import numpy 
import cv2
import operator
from matplotlib import pyplot as plt
import imageio

def read_from_file(file_object):
    img = imageio.imread(file_object, pilmode="RGB")
    return img

def distBetween(one,two):
    a = two[0] - one[0]
    b = two[1] - one[1]

    return numpy.sqrt((a**2)+(b**2))

def show_image(img):
    """Shows an image until any key is pressed"""
#    print(type(img))
#    print(img.shape)
#    cv2.imshow('image', img)  # Display the image
#    cv2.imwrite('images/gau_sudoku3.jpg', img)
#    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
#    cv2.destroyAllWindows()  # Close all windows
    return img

def scaleAndCentre(image,size, margin=0, background = 0):
    height, width = image.shape[:2]

    
    def centre(length):
        if length % 2 == 0:
            side1 = int((size-length) / 2)
            side2 = side1 
        else:
            side1 = int((size-length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale( r, x):
        return int(r*x)

    #if height>width:
        #t_pad = int(margin/2)
        #b_pad = t_pad 
        #ratio = (size-margin)/height
        #width , height = scale(ratio, width), scale(ratio, height)
        #t_pad, b_pad = centre(height)
    
    #else:
        #l_pad = int(margin/2)
        #r_pad = l_pad
        #ratio = (size - margin)/width
        #width, height = scale(ratio, width), scale(ratio, height)
        #t_pad, b_pad = centre(height)
    
    if height>width:
        t_pad = int(margin/2)
        b_pad=t_pad
        ratio = (size - margin)/height
        width, height = scale(ratio, width), scale(ratio,height)
        l_pad, r_pad = centre(width)
    else:
        l_pad = int(margin/2)
        r_pad = l_pad
        ratio = (size - margin)/width
        width, height = scale(ratio, width), scale(ratio,height)
        t_pad, b_pad = centre(height)
    
    image = cv2.resize(image,(width,height))
    image = cv2.copyMakeBorder(image, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(image,(size,size))


def displayFinal(numbers, colour=255):
    rows = []
    with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in numbers]
    for i in range(9):
        row = numpy.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
        rows.append(row)
    img = show_image(numpy.concatenate(rows))
    return img

#obtain the digit box from the whole square 
def extractNumber(image,rect, size):
    #obtain the digit box from the whole square 
    number = cut_from_rect(image,rect)
    
    #use fill feature finding to get the largest feature in the middle  of the box
    #margin used to define an area where we can expect to find a pixel from the number.
    height, width = number.shape[:2]
    margin = int(numpy.mean([height,width]) / 2.5)
    _, bbox, seed = findLargestFeature(number, [margin,margin], [width-margin, height-margin])
    number = cut_from_rect(number, bbox)

    #setup the number so its suitable for the ML process
    width = bbox[1][0] - bbox[0][0]
    height = bbox[1][1] - bbox[0][1]

    if width>0 and height>0 and (width*height) > 100 and len(number) > 0:
        return scaleAndCentre(number, size, 4)
    else:
        return numpy.zeros((size,size), numpy.uint8)

#obtain numbers from cells and build an array
def getNumber(image,squares, size):
    numbers = []
    image = processPuzzle(image.copy(), dilate=False)
    for square in squares:
        numbers.append(extractNumber(image,square,size))
    return numbers
       


def findLargestFeature(img, scan_tl= None, scan_br = None):
    imgCopy = img.copy()
    height, width = imgCopy.shape[:2]

    maxArea = 0
    seedPoint = (None,None)

    if scan_tl is None:
        scan_tl = [0,0]
    
    if scan_br is None:
        scan_br = [width,height]

    for x in range(scan_tl[0],scan_br[0]):
        for y in range(scan_tl[1],scan_br[1]):
            if img.item(y,x) == 255 and x<width and y<height: 
                area = cv2.floodFill(imgCopy, None, (x,y), 64)
                if area[0] > maxArea: 
                    maxArea = area[0]
                    seedPoint = (x,y)
    
    for x in range (width):
        for y in range(height):
            if imgCopy.item(y,x) == 255 and x<width and y<height:
                cv2. floodFill(imgCopy, None, (x,y) , 64)

    mask = numpy.zeros((height+2, width+2), numpy.uint8)

    if all([p is not None for p in seedPoint]):
        cv2.floodFill(imgCopy, mask,seedPoint, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range (width):
        for y in range(height):
            if imgCopy.item(y,x) == 64:
                cv2.floodFill(imgCopy,mask, (x,y), 0)
            
            if imgCopy.item(y,x) == 255:
                top = y if y<top else top
                bottom = y if y > bottom else bottom 
                left = x if x< left else left
                right = x if x > right else right
    bbox = [[left,top], [right,bottom]]

    return imgCopy, numpy.array(bbox, dtype= 'float32'), seedPoint

#processing the raw image to obtain only the sudoku puzzle
#invert the image colours
#and finally dilate
def processPuzzle(image, dilate=True):
    process = cv2.GaussianBlur(image.copy(), (9,9), 0)
    process = cv2.adaptiveThreshold(process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    process = cv2.bitwise_not(process,process)

    if dilate: 
        kernel = numpy.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],numpy.uint8)
        process = cv2.dilate(process, kernel)
    
    return process

#find all squares in the image and sort them in descending order of area and choose the largest one
def findCorners(image):
    contour, h = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = sorted(contour, key=cv2.contourArea, reverse=True)
    largestSquare = contour[0]

    br, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in largestSquare]), key=operator.itemgetter(1))
    tl, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in largestSquare]), key=operator.itemgetter(1))
    bl, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in largestSquare]), key=operator.itemgetter(1))
    tr, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in largestSquare]), key=operator.itemgetter(1))

    return [largestSquare[tl][0], largestSquare[tr][0], largestSquare[br][0], largestSquare[bl][0]]

#crop and return only the coordinates of the puzzle square
def cropAndWarp(image, contourCoord):
    tl, tr, br, bl = contourCoord[0], contourCoord[1], contourCoord[2], contourCoord[3]
    source = numpy.array([tl,tr,br,bl], dtype='float32')
    maxSide = max([distBetween(br,tr),
                   distBetween(tl,bl),
                   distBetween(br,bl),
                   distBetween(tl,tr)               
    ])

    dst = numpy.array([[0, 0], [maxSide - 1, 0], [maxSide - 1, maxSide - 1], [0, maxSide - 1]], dtype='float32')
    matrix = cv2.getPerspectiveTransform(source,dst)

    return cv2.warpPerspective(image, matrix, (int(maxSide), int(maxSide)))

def inferSquare(image):
    squares = []
    side = image.shape[:1]
    side = side[0]/9
    for j in range(9):
        for i in range(9):
            point1 = (i*side, j*side) 
            point2 = ((i+1)*side, (j+1)*side)
            squares.append((point1, point2))
    return squares

def cut_from_rect(img, rect):

	return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


def readPuzzle(image):
    #image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processImage = processPuzzle(image)
    cornersOfImage = findCorners(processImage)
    cropAndWarpImage = cropAndWarp(image, cornersOfImage)
    squares = inferSquare(cropAndWarpImage)
    numbers = getNumber(cropAndWarpImage, squares, 28)
    final = displayFinal(numbers)
    cv2.imshow('final',final)
    return final