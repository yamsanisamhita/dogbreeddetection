import numpy as np
import cv2

IMAGE_SIZE=(224,224)
IMAGE_FULL_SIZE=(224,224,3)

trainMyImageFolder="C:\\Users\\Samhi\\Desktop\\dogbreed\\train"

import pandas as pd

df=pd.read_csv('C:\\Users\\Samhi\\Desktop\\dogbreed\\labels.csv')
print(df.head())
print(df.describe())

grouplables=df.groupby("breed")["id"].count()
print(grouplables.head(10))

imgPath="C:\\Users\\Samhi\\Desktop\\dogbreed\\train\\00fa641312604199831755f96109fde7.jpg"
img=cv2.imread(imgPath)
cv2.imshow("img",img)
cv2.waitKey(0)

allImages=[]
allLabels=[]
import os

for ix,(image_name,breed) in enumerate(df[['id','breed']].values):
    img_dir=os.path.join(trainMyImageFolder,image_name+".jpg")
    print(img_dir)

    img=cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img=cv2.resize(img,IMAGE_SIZE,interpolation=cv2.INTER_AREA)
    allImages.append(resized_img)
    allLabels.append(breed)

print(len(allImages))
print(len(allLabels))

print("save the data")
np.save("C:\\Users\\Samhi\\Desktop\\dogbreed\\allDogImages.npy",allImages)
np.save("C:\\Users\\Samhi\\Desktop\\dogbreed\\allDogLables.npy",allLabels)
print("finish save the data")



