from tabnanny import verbose
from turtle import pu
import numpy
import cv2
import matplotlib.pyplot as plot
from keras.models import model_from_json


json_file =open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loadedModel = model_from_json(loaded_model_json)
loadedModel.load_weights('model/model.h5')
print("Loaded saved model from disk.")

def predictNumber(image):
    imageResize = cv2.resize(image,(28,28))
    imageResizeCopy = imageResize.reshape(1, 1, 28, 28)
    #loadedModelPred = loadedModel.predict_classes(imageResizeCopy, verbose=0)
    loadedModelPred = numpy.argmax(loadedModel.predict(imageResizeCopy), axis=1)
    return loadedModelPred[0]

def extract(puzzle):
    puzzle = cv2.resize(puzzle, (450,450))

    grid = numpy.zeros([9,9])
    for i in range(9):
        for j in range(9): 
            image = puzzle[i*50:(i+1)*50,j*50:(j+1)*50]
            if image.sum()>25000:
                grid[i][j] = predictNumber(image)
            else:
                grid[i][j] =0;
    return grid.astype(int)