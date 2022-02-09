# DNN had to have pictures with only the clothing in it and it also had to be centered, which is not very practical
# Convolution is a filter that passes over an image, processes it, and extracts the important features
# We extract it by getting only the important features and blur the others (aka feature mapping)

import cv2 as cv
import numpy as np
from scipy import misc
img = misc.ascent()

# Just showing the img in grayscale
import matplotlib.pyplot as plt
# plt.grid(False)
# plt.gray()
# plt.axis('off')
# plt.imshow(img)
# plt.show()

# Copy the img
img_transformed = np.copy(img)
# Getting dimensions
size_x = img_transformed.shape[0]
size_y = img_transformed.shape[1]

# Creating the 3x3 kernel
# This filter detects edges nicely
# It creates a filter that only passes through sharp edges and straight lines. 
# Experiment with different values for fun effects.
# filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]] 
# A couple more filters to try for fun!
# filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]
filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
# If all the digits in the filter don't add up to 0 or 1, you 
# should probably do a weight to get it to do so
# so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1 if you want to normalize them
weight  = 1

# Blurring the img
for x in range(1,size_x-1):
  for y in range(1,size_y-1):
      output_pixel = 0.0
      output_pixel = output_pixel + (img[x - 1, y-1] * filter[0][0])
      output_pixel = output_pixel + (img[x, y-1] * filter[0][1])
      output_pixel = output_pixel + (img[x + 1, y-1] * filter[0][2])
      output_pixel = output_pixel + (img[x-1, y] * filter[1][0])
      output_pixel = output_pixel + (img[x, y] * filter[1][1])
      output_pixel = output_pixel + (img[x+1, y] * filter[1][2])
      output_pixel = output_pixel + (img[x-1, y+1] * filter[2][0])
      output_pixel = output_pixel + (img[x, y+1] * filter[2][1])
      output_pixel = output_pixel + (img[x+1, y+1] * filter[2][2])
      output_pixel = output_pixel * weight
      if(output_pixel<0):
        output_pixel=0
      if(output_pixel>255):
        output_pixel=255
      img_transformed[x, y] = output_pixel

# Plot the image. Note the size of the axes -- they are 512 by 512
# plt.gray()
# plt.grid(False)
# plt.imshow(img_transformed)
# #plt.axis('off')
# plt.show()   

# Pooling greatly helps with detecting features 
# Pooling layers reduce the overall amount of information in an image while maintaining the features that are detected as present

# Max Pooling
# Iterate over the image and, at each point, consider the pixel and its immediate neighbors to the right, beneath, and right-beneath
# Largest of those are loaded into the new img -> image becomes 1/4 size
# (2, 2) Max Pooling
new_x = int(size_x/2)
new_y = int(size_y/2)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
  for y in range(0, size_y, 2):
    pixels = []
    pixels.append(img_transformed[x, y])
    pixels.append(img_transformed[x+1, y])
    pixels.append(img_transformed[x, y+1])
    pixels.append(img_transformed[x+1, y+1])
    pixels.sort(reverse=True)
    newImage[int(x/2),int(y/2)] = pixels[0]
 
# Plot the image. Note the size of the axes -- now 256 pixels instead of 512
# You can actually see that the key features were enhanced despite being in lower resolution (less data)
plt.gray()
plt.grid(False)
plt.imshow(newImage)
#plt.axis('off')
plt.show()
