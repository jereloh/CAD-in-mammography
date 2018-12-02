'''
01_Load_Images 
[Goal]
Learn how to:
1. Read Images from a given path
2. Display Images from a given path
3. Resize Images in the Display port
4. Quit Display Port
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

# Create window for image, to keep image size within current display size
height, width, channels = image.shape
cv2.namedWindow("Viewport", cv2.WINDOW_NORMAL)
height //= 4
width //= 4
cv2.resizeWindow("Viewport", width , height) 

# view image
cv2.imshow("Viewport", image)
#cv2.waitKey(0)
# Improvise waitKey to only close upon pressing esc
k = cv2.waitKey(0)
while k != 27:
  k = cv2.waitKey(0)

cv2.destroyAllWindows()
