import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps
x,y = fetch_openml("mnist_784", version = 1, return_X_y = True)
xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size = 7500, test_size = 2500)
xTrainScaled = xTrain/255.0
xTestScaled = xTest/255.0
lr = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(xTrainScaled, yTrain)
def getPrediction(image):
    im_pil = Image.open(image)
    imagebw = im_pil.convert("L")
    imagebwresized = imagebw.resize((28,28), Image.ANTIALIAS)
    pixel_filter = 20
    minpixel = np.percentile(imagebwresized, pixel_filter)
    imageinverted = np.clip(imagebwresized-minpixel, 0, 255)
    maxpixel = np.max(imagebwresized)
    imageinverted = np.asarray(imageinverted)/maxpixel
    testSample = np.array(imageinverted).reshape(1,784)
    testPredict = lr.predict(testSample)
    return testPredict[0]
