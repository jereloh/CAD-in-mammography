'''
dicom converter to png
[Goal]
1. Using pydicom, py module for dicom read dicom, png using pypng 
2. convert to png using opencv etc.
    2.1 toss dicom to numpy array
    2.2 toss numpy array into jpeg
3. file sort
4. sort from .csv (benign, malignant etc.)
'''

# import necessary packages
import os
import pydicom, png
import numpy as np
import argparse

# construct argument parsing
argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--input", "-o", "--output", required=True,
	help="please insert input and output folder path of the image")
args = vars(argparser.parse_args())

def dicom2png(source_folder, output_folder): #Ref https://github.com/pydicom/pydicom/issues/352
    list_of_files = os.listdir(source_folder)
    for file in list_of_files:
        try:
            ds = pydicom.dcmread(os.path.join(source_folder,file))
            shape = ds.pixel_array.shape

            # Convert to float to avoid overflow or underflow losses.
            image_2d = ds.pixel_array.astype(float)

            # Rescaling grey scale between 0-255
            image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

            # Convert to uint
            image_2d_scaled = np.uint8(image_2d_scaled)

            # Write the PNG file
            with open(os.path.join(output_folder,file)+'.png' , 'wb') as png_file:
                w = png.Writer(shape[1], shape[0], greyscale=True)
                w.write(png_file, image_2d_scaled)
        except:
            print('Could not convert: ', file)


dicom2png(args["input"], args["output"])

