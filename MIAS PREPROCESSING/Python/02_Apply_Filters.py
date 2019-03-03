'''
01_Apply_Filters
[Goal]
Learn how to:
1. Add filters
2. Display filtered and unfiltered images
'''

# import necessary packages
import argparse
import cv2
 
# construct argument parsing
argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--image", required=True,
	help="please insert path of the image")
args = vars(argparser.parse_args())

# Load image
image = cv2.imread(args["image"])

# Adding Filters 
thresh = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)[1]

# Create window for image, to keep image size within current display size
height, width, channels = image.shape
height //= 4
width //= 4
cv2.namedWindow("Viewport", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Viewport", width , height) 
cv2.namedWindow("Viewport1", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Viewport1", width , height) 

# view image
cv2.imshow("Viewport", image)
cv2.imshow("Viewport1", thresh)
# Improvise waitKey to only close upon pressing esc
k = cv2.waitKey(0)
while k != 27:
  k = cv2.waitKey(0)

cv2.destroyAllWindows()
