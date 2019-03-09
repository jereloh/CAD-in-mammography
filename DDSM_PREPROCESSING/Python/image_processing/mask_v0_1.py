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
import numpy as np# Display multiple images
 

def maskImages(inputIM, outputIM):
    # Load image
    image = cv2.imread(inputIM)
    
    # Copy original image for viewing purposes
    imageInput = image.copy()

    # Gausian blurring before finding contours
    image_Gauzz = cv2.GaussianBlur(image,(5,5),0)

    # Adding Filters 
    gray = cv2.cvtColor(image_Gauzz, cv2.COLOR_BGR2GRAY) #Added this to allow contours to work
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

    # Finding Contours
    contours= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] # only need outer contours so we use RETR_EXTERNAL

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
            cv2.fillPoly(imageFinal, pts = [cntUnwanted], color = (0,0,0))
    
    # Testing, don't write first!
    cv2.imwrite(outputIM,imageFinal)
    '''
    # view image
    numpy_viewport = np.concatenate((imageInput,image_Gauzz, image, imageFinal),axis = 1)
    cv2.imshow("Viewport", numpy_viewport)
    # Improvise waitKey to only close upon pressing esc
    k = cv2.waitKey(0)
    while k != 27:
        k = cv2.waitKey(0)

    cv2.destroyAllWindows()
    '''

# construct argument parsing
argparser = argparse.ArgumentParser()
argparser.add_argument("-f", "--folder", required=True,
	help="please insert path of the folder containing the images")
args = vars(argparser.parse_args())

osDIR = os.listdir(args["folder"])

# For progress bar
i = 1
for toConvert in osDIR:
    if toConvert.endswith(".png"):
        if not os.path.exists(os.path.join(args["folder"],'Masked')):
            os.mkdir(os.path.join(args["folder"],'Masked'))
            print ("Created")
        print ("[",i,"/",len(osDIR),"]",end= '')
        i +=1 
        print (os.path.join(args["folder"],"Masked",toConvert))
        maskImages((os.path.join(args["folder"],toConvert)),(os.path.join(args["folder"],"Masked",toConvert)))
        #print ("Converted:"+os.path.join(args["folder"],"Masked",toConvert))
        