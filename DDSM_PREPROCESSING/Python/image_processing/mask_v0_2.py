'''
06_Final adapted
[Goal]
Learn how to:
1. Create a function from 05_Save_AOI
2. Automate generation of images without tags into another folder
3. Tweak tresh parameters
'''

# import necessary packages

import argparse, os, glob, time
import cv2
import concurrent.futures
from tqdm import tqdm
import multiprocessing
 
# [Function] masking Images script
def maskImages(inputIM):
    # Load image
    image = cv2.imread(inputIM)
    
    # Copy original image for viewing purposes
    imageInput = image.copy()

    '''
    new image_wBorder to create border line 34
    '''
    # Create border? https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html
    # Initialize arguments for the filter
    image_wBorder = cv2.rectangle(imageInput, (0, 0), (image.shape[1],image.shape[0]), (0,0,0), 120)

    # Gausian blurring before finding contours
    image_Gauzz = cv2.GaussianBlur(image_wBorder,(5,5),0)

    # Adding Filters 
    gray = cv2.cvtColor(image_Gauzz, cv2.COLOR_BGR2GRAY) #Added this to allow contours to work
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

    # Finding Contours
    contours= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] 
    # only need outer contours so we use RETR_EXTERNAL

    # Display contour
    imageFinal = image_wBorder.copy()
    
    cv2.drawContours(image_wBorder, contours, -1, (0,255,255), 3)

    # Filter Contours Area
    cntArea = [ ]
    for cnt in contours:
        cntArea.append(cv2.contourArea(cnt))

    for cntUnwanted in contours:
        if cv2.contourArea(cntUnwanted) != max(cntArea):
        # Fill unwanted contours black
            cv2.fillPoly(imageFinal, pts = [cntUnwanted], color = (0,0,0))
    
    # Prep Write file and check if UNIX or not
    if os.name == 'posix':
        toConvert_filename = inputIM.split("/").pop()
    elif os.name == 'nt':
        toConvert_filename = inputIM.split("\\").pop()
    else:
        print ("Operating System not supported: "+os.name)
        quit()
    # Write file
    cv2.imwrite(os.path.join(args["folder"],"Masked",toConvert_filename),imageFinal)
    # View images for troubleshooting
    import numpy as np# Display multiple images
    # view image
    numpy_viewport = np.concatenate((image, image_wBorder, imageFinal),axis = 1)
    #numpy_viewport = np.concatenate((image_wBorder),axis = 1)
    #Call Viewpoert
    cv2.namedWindow("Viewport", cv2.WINDOW_NORMAL)
    # Resize window size
    cv2.resizeWindow("Viewport",1280,720)
    cv2.imshow("Viewport", numpy_viewport)
    # Improvise waitKey to only close upon pressing esc
    k = cv2.waitKey(0)
    while k != 27:
        k = cv2.waitKey(0)

    cv2.destroyAllWindows()

    # for visualization of the loading bar
    time.sleep(0.001)

# [Function] Create necessary folder function
def chkFolder(flder_path):
    if not os.path.exists(flder_path):
        print ("Folder don't exist. Making..")
        os.makedirs(flder_path)

# construct argument parsing
argparser = argparse.ArgumentParser()
argparser.add_argument("-f", "--folder", required=True,
	help="please insert path of the folder containing the images")
args = vars(argparser.parse_args())

chkFolder(args["folder"]+"Masked")

files = glob.glob( os.path.join(args["folder"],'*.png'), recursive=True)
maskImages(files[0])
#For trouble shooting 
'''
# An attempt at parallel processing
with concurrent.futures.ThreadPoolExecutor(multiprocessing.cpu_count()) as executor: 
    list(tqdm(executor.map(maskImages, files), total=len(files)))
    #executor.map(maskImages, files)
'''