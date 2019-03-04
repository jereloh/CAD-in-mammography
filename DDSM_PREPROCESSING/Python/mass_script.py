'''
Mass_DDSM script dicom converter and sorter in python
[Goal]
1. Read .csv file provided, to be placed within the input folder
2. Read folder name which correlates to patient
3. Toss patient case into folders benign, malignant, normal
'''

# Call the necessary imports
import os
import pydicom, png
import numpy
import argparse

# construct argument parsing
argparser = argparse.ArgumentParser()
argparser.add_argument( "--csv", required=True,
	help="please insert input and output folder path of the image and insert the appropriate csv file name")
args = vars(argparser.parse_args())

def dicom2png(src_folder, out_folder): #Ref https://github.com/pydicom/pydicom/issues/352
    list_of_files = os.listdir(src_folder)
    for file in list_of_files:
        try:
            ds = pydicom.dcmread(os.path.join(src_folder,file))
            shape = ds.pixel_array.shape

            # Convert to float to avoid overflow or underflow losses.
            image_2d = ds.pixel_array.astype(float)

            # Rescaling grey scale between 0-255
            image_2d_scaled = (numpy.maximum(image_2d,0) / image_2d.max()) * 255.0

            # Convert to uint
            image_2d_scaled = numpy.uint8(image_2d_scaled)

            # Write the PNG file
            with open(os.path.join(out_folder,file)+'.png' , 'wb') as png_file:
                w = png.Writer(shape[1], shape[0], greyscale=True)
                w.write(png_file, image_2d_scaled)
        except:
            print('Could not convert: ', file)

'''
readCSV
[Goal]: Read file into an matrix to correlate the folder names to the correct status of the images
'''
def readCSV(src_folder):
    #read csv
    corr_csv = numpy.genfromtxt(args["csv"], delimiter=',', names=True)
    print (corr_csv[0])

readCSV(args["csv"])


#dicom2png(args["input"], args["output"])