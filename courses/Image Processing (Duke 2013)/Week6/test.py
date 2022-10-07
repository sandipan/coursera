import numpy as np
import cv2
import matplotlib.pylab as plt

# Load an color image in grayscale
img = cv2.imread('messi5.jpg',0)
print img.shape
#plt.imshow(img)
#plt.plot(img)
#plt.show()

(w, h) = img.shape
print w, h
for x in range(w):
	for  y in range(h): 
	   Ix = 0
	   Iy = 0 
  	   if x < w - 1:
  	      Ix = img[x+1,y] - img[x,y];
 	   if y < h - 1:
  	      Iy = img[x,y+1] - img[x,y];   
	   img[x,y] += abs(np.sqrt(Ix**2 + Iy**2))
plt.imshow(img, cmap='gray')
#plt.plot(img)
plt.show()
