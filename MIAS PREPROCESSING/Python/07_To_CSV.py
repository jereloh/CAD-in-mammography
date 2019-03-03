'''
07_To_CSV
[Goal]
Learn how to:
1. Convert image to .csv
'''
'''
05_Save_AOI
[Goal]
Learn how to:
1. Omit all unwanted area to be black
2. Save final Image in a file
'''

# import necessary packages

import argparse
import cv2 
import numpy
 
# construct argument parsing
argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--image", required=True,
	help="please insert path of the image")
args = vars(argparser.parse_args())

# Load image
image = cv2.imread(args["image"])

# Adding Filters 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Make sure it is 0-255

height, width, channels = image.shape
height //= 4
width //= 4
cv2.namedWindow("Viewport", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Viewport", width , height)
cv2.imshow("Viewport", gray)

numpy.savetxt("converted.csv", gray, fmt='%d', delimiter=',',newline='\n')

#cv2.imwrite("new.jpg",imageFinal)
# Improvise waitKey to only close upon pressing esc
k = cv2.waitKey(0)
while k != 27:
  k = cv2.waitKey(0)

cv2.destroyAllWindows()
