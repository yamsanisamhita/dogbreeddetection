import tensorflow as tf
import os
import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input

modelFile="C:\\Users\\Samhi\\Desktop\\dogbreed\\dog_breed_model.h5"
model=tf.keras.models.load_model(modelFile)

inputShape=(224,224)

allLabes=np.load("C:\\Users\\Samhi\\Desktop\\dogbreed\\allDoglables.npy")
categories=np.unique(allLabes)

def prepareImage(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized=cv2.resize(img_rgb,inputShape,interpolation=cv2.INTER_AREA)
    imgResult=np.expand_dims(resized,axis=0)
    imgResult = preprocess_input(imgResult)
    return imgResult 

testImagesPath="C:\\Users\\Samhi\\Desktop\\dogbreed\\test\\00a3edd22dc7859c487a64777fc8d093.jpg"

img=cv2.imread(testImagesPath)
imageForModel=prepareImage(img)

resultArray=model.predict(imageForModel)
answers=np.argmax(resultArray,axis=1)
print(answers)

text=categories[answers[0]]
print("Predicted Label: ",text) 

