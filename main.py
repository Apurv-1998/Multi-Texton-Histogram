# importing libraries
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def avg(a,b,c):
	return 16*a+4*b+c

# Start Feeding The Image Into Database


from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017')
path = "C:\\Users\\hp\\Desktop\\MultiTexton\\Corel\\"
Database = []
for pre in range(0,1):
	prefix = str(pre)+"_"
	print(prefix)
	for entry in range(0,100):
	    imagename = path + prefix + str(entry)+'.jpg'
	    print(imagename)
	    img = cv2.imread(imagename)
	    width, height, channels = img.shape



	    #Texture Orientation Detection


	    CSA = 64 #Color Quantization Vectors
	    CSB = 18 #Texture Orientation Vectors
	    arr = np.zeros(3*width*height).reshape(width,height,3)
	    ori = np.zeros(width * height).reshape(width, height)



	    gxx = gyy = gxy = 0.0 #gradients along x and y direction
	    rh = gh = bh = 0.0 #gradient of red,green and blue along horizontal
	    rv = gv = bv = 0.0 #gradient of red,green and blue along vertical
	    theta = np.zeros(width*height).reshape(width,height)

	    for i in range(1, width-2):
	        for j in range(1, height-2):
	            rh=arr[i-1,j+1,0] + 2*arr[i,j + 1,0] + arr[i+1, j+1,0] - (arr[i-1, j - 1, 0] + 2 * arr[i,j-1, 0] + arr[i + 1, j - 1, 0]) #calculation Using Sobel
	            gh=arr[i-1,j+1,1] + 2*arr[i,j + 1,1] + arr[i+ 1,j+1,1] - (arr[i-1, j - 1, 1] + 2 * arr[i,j-1, 1] + arr[i + 1, j - 1, 1])
	            bh=arr[i-1,j+1,2] + 2*arr[i,j + 1,2] + arr[i+ 1,j+1,2] - (arr[i-1, j - 1, 2] + 2 * arr[i,j-1, 2] + arr[i + 1, j - 1, 2])
	            rv=arr[i+1,j-1,0] + 2*arr[i+1, j, 0] + arr[i+ 1,j+1,0] - (arr[i-1, j - 1, 0] + 2 * arr[i-1,j, 0] + arr[i - 1, j + 1, 0])
	            gv=arr[i+1,j-1,1] + 2*arr[i+1, j, 1] + arr[i+ 1,j+1,1] - (arr[i-1, j - 1, 1] + 2 * arr[i-1,j, 1] + arr[i - 1, j + 1, 1])
	            bv=arr[i+1,j-1,2] + 2*arr[i+1, j, 2] + arr[i+ 1,j+1,2] - (arr[i-1, j - 1, 2] + 2 * arr[i-1,j, 2] + arr[i - 1, j + 1, 2])
	            
	            gxx = math.sqrt(rh * rh + gh * gh + bh * bh) #final Gradient Calculation of x-component
	            gyy = math.sqrt(rv * rv + gv * gv + bv * bv) #final Gradient Calculation of y-component
	            gxy = rh * rv + gh * gv + bh * bv #final gradient component for co-occurnece
	            
	            theta[i,j] = (math.acos(gxy / (gxx * gyy + 0.0001))*180 / math.pi)#cos inverse



	    #Color Quantization

	    ImageX = np.zeros(width * height).reshape(width, height)


	    R = G = B = 0 # Red, Green and Blue Components of the image
	    VI = SI = HI = 0 
	    for i in range(0, width):
	        for j in range(0, height):
	            R = img[i,j][0]
	            G = img[i,j][1]
	            B = img[i,j][2]
	            
	            if (R >=0 and R <= 64):
	                VI = 0;
	            if (R >= 65 and R <= 128):
	                VI = 1;
	            if (R >= 129 and R <= 192):
	                VI = 2;
	            if (R >= 193 and R <= 255):
	                VI = 3;
	            if (G>= 0 and G <= 64):
	                SI = 0;
	            if (G >= 65 and G <= 128):
	                SI = 1;
	            if (G >= 129 and G <= 192):
	                SI = 2;
	            if (G >= 193 and G <= 255):
	                SI = 3;
	            if (B >= 0 and B <= 64):
	                HI = 0;
	            if (B >= 65 and B <= 128):
	                HI = 1;
	            if (B >= 129 and B <= 192):
	                HI = 2;
	            if (B >= 193 and B <= 255):
	                HI = 3;
	            
	            ImageX[i, j] = avg(VI,SI,HI)


	    for i in range(0, width):
	        for j in range(0, height):
	            ori[i,j] = round(theta[i,j]*CSB/180) #step size of 10 since CSB = 18
	            
	            if(ori[i,j]>=CSB-1):
	                ori[i,j]=CSB-1    


	    #Texton Detection

	    Texton = np.zeros(width * height).reshape(width, height)

	    for i in range(0,(int)(width/2)):
	        for j in range(0,(int)(height/2)):
	            if(ImageX[2*i,2*j] == ImageX[2*i+1,2*j+1]):    #Texton for left diagonal
	                Texton[2 * i, 2 * j] = ImageX[2 * i, 2 * j];
	                Texton[2 * i + 1, 2 * j] = ImageX[2 * i + 1, 2 * j];
	                Texton[2 * i, 2 * j + 1] = ImageX[2 * i, 2 * j + 1];
	                Texton[2 * i + 1, 2 * j + 1] = ImageX[2 * i + 1, 2 * j + 1];
	            
	            if (ImageX[2*i,2*j+1] == ImageX[2*i+1,2*j]):    #Texton for right diagonal
	                Texton[2 * i, 2 * j] = ImageX[2 * i, 2 * j];
	                Texton[2 * i + 1, 2 * j] = ImageX[2 * i + 1, 2 * j];
	                Texton[2 * i, 2 * j + 1] = ImageX[2 * i, 2 * j + 1];
	                Texton[2 * i + 1, 2 * j + 1] = ImageX[2 * i + 1, 2 * j + 1];
	            
	            if (ImageX[2*i,2*j] == ImageX[2*i+1,2*j]):  #Texton for vertical
	                Texton[2 * i, 2 * j] = ImageX[2 * i, 2 * j];
	                Texton[2 * i + 1, 2 * j] = ImageX[2 * i + 1, 2 * j];
	                Texton[2 * i, 2 * j + 1] = ImageX[2 * i, 2 * j + 1];
	                Texton[2 * i + 1, 2 * j + 1] = ImageX[2 * i + 1, 2 * j + 1];
	                
	            if (ImageX[2*i,2*j] == ImageX[2*i,2*j+1]): #Texton for horizontal
	                Texton[2 * i, 2 * j] = ImageX[2 * i, 2 * j];
	                Texton[2 * i + 1, 2 * j] = ImageX[2 * i + 1, 2 * j];
	                Texton[2 * i, 2 * j + 1] = ImageX[2 * i, 2 * j + 1];
	                Texton[2 * i + 1, 2 * j + 1] = ImageX[2 * i + 1, 2 * j + 1];                   


	    # Multi-Texton Histogram


	    MatrixH = np.zeros(CSA + CSB).reshape(CSA + CSB)
	    MatrixV = np.zeros(CSA + CSB).reshape(CSA + CSB)
	    MatrixRD = np.zeros(CSA + CSB).reshape(CSA + CSB)
	    MatrixLD = np.zeros(CSA + CSB).reshape(CSA + CSB)

	    D = 1 #distance parameter i.e. step-size

	    for i in range(0, width):
	        for j in range(0, height-D):
	            if(ori[i, j+D] == ori[i, j]):
	                MatrixH[(int)(Texton[i,j])] += 1;
	            if(Texton[i, j + D] == Texton[i, j]):
	                MatrixH[(int)(CSA + ori[i, j])] += 1;

	    for i in range(0, width-D):
	        for j in range(0, height):
	            if(ori[i + D, j] == ori[i, j]):
	                MatrixV[(int)(Texton[i,j])] += 1;
	            if(Texton[i + D, j] == Texton[i, j]):
	                MatrixV[(int)(CSA + ori[i, j])] += 1;

	    for i in range(0, width-D):
	        for j in range(0, height-D):
	            if(ori[i + D, j + D] == ori[i, j]):
	                MatrixRD[(int)(Texton[i,j])] += 1;
	            if(Texton[i + D, j + D] == Texton[i, j]):
	                MatrixRD[(int)(CSA + ori[i, j])] += 1;
	                
	    for i in range(D, width):
	        for j in range(0, height-D):
	            if(ori[i - D, j + D] == ori[i, j]):
	                MatrixLD[(int)(Texton[i,j])] += 1;
	            if(Texton[i - D, j + D] == Texton[i, j]):
	                MatrixLD[(int)(CSA + ori[i, j])] += 1;

	    
	    #Final MTH Feature Vector

	    MTH = np.zeros(CSA + CSB).reshape(CSA + CSB)

	    for i in range(0, CSA + CSB):
	        MTH[i] = ( MatrixH[i] + MatrixV[i] + MatrixRD[i] + MatrixLD[i])/4.0

	    
	    Database.append(MTH)
	    entry+=1
	    print("Entered for "+imagename)


#Plotting the Histogram for visualization purpose
print(len(MTH))

plt.axis([0,82,0,4000])
plt.bar(np.arange(82),MTH)
plt.xlabel('Bin Size')
plt.ylabel('Frequency')
plt.title('Histogram of MTH')
plt.grid(True)

plt.show()


#Showing the database values
Database = np.array(Database)
collection = client.MTH.coralTest
collection.insert({"distances":Database.tolist(),"name":'Coral Dataset'})
print ("Stored Values in the Database ",Database)

