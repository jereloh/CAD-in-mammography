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
import numpy as np

 
# construct argument parsing
argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--image", required=True,
	help="please insert path of the image")
args = vars(argparser.parse_args())

# Load image
image = cv2.imread(args["image"])

# Adding Filters 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Added this to allow contours to work
thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

# Finding Contours
contours= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] 
# only need outer contours so we use RETR_EXTERNAL
# Display contour
imageFinal = image.copy()

cv2.drawContours(image, contours, -1, (0,255,255), 3)

# Filter Contours Area
cntArea = [ ]
for cnt in contours:
    cntArea.append(cv2.contourArea(cnt))

for cntUnwanted in contours:
    if cv2.contourArea(cntUnwanted) != max(cntArea):
      # Fill unwanted contours black
      cv2.fillPoly(imageFinal, pts = [cntUnwanted], color = (255,0,255))

# Create window for image, to keep image size within current display size
height, width, channels = image.shape
height //= 4
width //= 4
cv2.namedWindow("Viewport", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Viewport", width , height) #cv2.fillPoly(imageFinal,cntUnwanted, color=(255,255,255))

cv2.namedWindow("Viewport1", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Viewport1", width , height) 
cv2.namedWindow("Viewport2", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Viewport2", width , height) 

# view image
cv2.imshow("Viewport", image)
cv2.imshow("Viewport1", thresh)
cv2.imshow("Viewport2", imageFinal)
#cv2.imwrite("new.jpg",imageFinal)
# Improvise waitKey to only close upon pressing esc
k = cv2.waitKey(0)
while k != 27:
  k = cv2.waitKey(0)

cv2.destroyAllWindows()
