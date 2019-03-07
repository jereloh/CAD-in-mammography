'''
06_Final
[Goal]
Learn how to:
1. Create a function from 05_Save_AOI
2. Automate generation of images without tags into another folder
3. Tweak tresh parameters
'''

# import necessary packages

import argparse
import cv2
import os
 

def maskImages(inputIM, outputIM):
    # Load image
    image = cv2.imread(inputIM)

    # Adding Filters 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Added this to allow contours to work
    thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]

    # Finding Contours
    contours= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] # only need outer contours so we use RETR_EXTERNAL
    # Display contour
    imageFinal = image.copy()
    
    cv2.drawContours(image, contours, -1, (0,255,255), 3)

    cv2.imshow("image",image)

    # Filter Contours Area
    cntArea = [ ]
    for cnt in contours:
        cntArea.append(cv2.contourArea(cnt))

    for cntUnwanted in contours:
        if cv2.contourArea(cntUnwanted) != max(cntArea):
        # Fill unwanted contours black
            cv2.fillPoly(imageFinal, pts = [cntUnwanted], color = (0,0,0))

    '''
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
    '''
    print(outputIM)
    cv2.imwrite(outputIM,imageFinal)
    
    # Improvise waitKey to only close upon pressing esc
    k = cv2.waitKey(0)
    while k != 27:
        k = cv2.waitKey(0)

    cv2.destroyAllWindows()
    

# construct argument parsing
argparser = argparse.ArgumentParser()
argparser.add_argument("-f", "--folder", required=True,
	help="please insert path of the folder containing the images")
args = vars(argparser.parse_args())

osDIR = os.listdir(args["folder"])
print(osDIR)

for toConvert in osDIR:
    if toConvert.endswith(".png"):
        if not os.path.exists(os.path.join(args["folder"],'Masked')):
            os.mkdir(os.path.join(args["folder"],'Masked'))
            print ("Created")
        #print (os.path.join(args["folder"],"Masked",toConvert))
        maskImages((os.path.join(args["folder"],toConvert)),(os.path.join(args["folder"],"Masked",toConvert)))
        #print ("Converted:"+os.path.join(args["folder"],"Masked",toConvert))