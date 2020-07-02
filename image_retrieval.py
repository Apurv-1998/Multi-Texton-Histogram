#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import operator
from IPython.display import Image
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#Establishing connection with the mongo client

from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017')
collection = client.MTH.coralTest


#taking input from the user


db = []
path = "C:\\Users\\hp\\Desktop\\MultiTexton\\Corel\\"
num = np.int64(input('Enter the input Image: '))
for x in collection.find():
    db = np.array(x['distances'])
print(path +str(0)+"_"+str(num)+'.jpg')
Image(path +str(0)+"_"+str(num)+'.jpg')



print(len(db))

inputImage = db[int(num)-1]


#calculating the distances between the query image and the incoming image from the database
distance = np.zeros(100*82).reshape(100,82)
for i in range(100):
    for j in range(82):
        distance[i,j] = abs(db[i,j] - inputImage[j])/(1 + db[i,j] + inputImage[j]) #distance metric

distanceSum = np.sum(distance,axis=1)


#storing the incoming image and the distance as a (key,value) pair.

keys = np.arange(len(distanceSum),dtype=int)

Imagedictionary = dict(zip(keys, distanceSum))

#showing the image in a sorted manner
sorted_images = sorted(Imagedictionary.items(), key=operator.itemgetter(1))




# Displaying all the images
def display_photo(path):
    print("File: {}".format(path))
    img = mpimg.imread(path)
    imgplot = plt.imshow(img)
    plt.show()



i = 0;
Resultimages = []
ResultHists = np.zeros(1000*82).reshape(1000,82)
for key in sorted_images:
    if(i<=20):
        print("key ",key)
        ResultHists[i]=db[key[0]]
        i = i +1
        imageName = path +str(0)+"_"+str(key[0])+'.jpg'
        print (imageName)
        display_photo(imageName)
    else:
        break;





