'''
Mass_DDSM script dicom converter and sorter in python
[Goal]
1. Read .csv file provided, to be placed within the input folder
2. Read folder name which correlates to patient
3. Toss patient case into folders benign, malignant, normal
'''

# Call the necessary imports
import os, argparse
import pydicom, png
import numpy
import pandas
import glob

# construct argument parsing
argparser = argparse.ArgumentParser()
argparser.add_argument( "-i", "--input", required=True,
	help="please insert input and output folder path of the image and insert the appropriate csv file name")
args = vars(argparser.parse_args())

def dicom2png(src, out_folder, file_name): #Ref https://github.com/pydicom/pydicom/issues/352
    try:
        #ds = pydicom.dcmread(os.path.join(src_folder,file))
        ds = pydicom.dcmread(src)
        shape = ds.pixel_array.shape

        # Convert to float to avoid overflow or underflow losses.
        image_2d = ds.pixel_array.astype(float)

        # Rescaling grey scale between 0-255
        image_2d_scaled = (numpy.maximum(image_2d,0) / image_2d.max()) * 255.0

        # Convert to uint
        image_2d_scaled = numpy.uint8(image_2d_scaled)

        # Write the PNG file
        with open(os.path.join(out_folder,file_name)+'.png' , 'wb') as png_file:
            w = png.Writer(shape[1], shape[0], greyscale=True)
            w.write(png_file, image_2d_scaled)
        print(file_name+".png","converted")
    except:
        print('Could not convert: ',src)

'''
[Function] readCSV
[Goal]: Read file into an dataframe to correlate the folder names to the correct status of the images
'''
def readCSV(data_path):
    #read csv
    corr_df =  pandas.read_csv(data_path)
    # create new dataframe that populates patient id and 
    # We understand that the folder is named as such .../CBIS_DDSM/Mass-Training_P_00001_LEFT_MLO/...(file contents)
    # scrub dataframe
    return corr_df[['patient_id','left or right breast','image view','pathology']]
'''
[Function] Create folders
'''
def chkFolder(flder_path):
    if not os.path.exists(flder_path):
        print ("Folder don't exist. Making..")
        os.makedirs(flder_path)
'''
[Function] Subfolder finder 
'''
def dcmFullPath(flder_path):
    # Find all subfolders using glob module, iterate through and do sth based on file path name
    files = glob.glob(flder_path + '/*/*/*.dcm', recursive=True)
    if (len(files) == 1):
        return files[0]
    else:
        print ("More than 1 file found", files)
        quit()
'''
"Main"
'''
# readCSV, put into dataframe
corr_df_datapath = args["input"]+ "mass_case_description_train_set.csv"
print (corr_df_datapath)
corr_df = readCSV(corr_df_datapath)
print (corr_df)
# Idea: due to space constrain we won't be modifying the raw data files 
# Before trying anything here, we need to find out the file to be converted belongs to which patient and its pathology

# Check folders Malignant and Benign created
chkFolder(args["input"]+"BENIGN")
chkFolder(args["input"]+"MALIGNANT")
chkFolder(args["input"]+"OTHERS")

# Iterate through corr_df 
for index, row in corr_df.iterrows():

    # Place the parse string together? search the file path for it.
    # Generate File interested file path from .csv
    str_parse = row["patient_id"]+"_"+row["left or right breast"]+"_"+row["image view"] 
    str_file_path = args["input"]+"CBIS-DDSM/Mass-Training_"+str_parse+"/"
    # Generate file path of .dcm
    input_Dicom2PNG = dcmFullPath(str_file_path)

    # Sort input into relevant folders
    if "BENIGN" in row["pathology"] :  
        output_Dicom2PNG = (args["input"]+"BENIGN")      
    elif "MALIGNANT" in row["pathology"] :      
        output_Dicom2PNG = (args["input"]+"MALIGNANT")
    else:
        output_Dicom2PNG = (args["input"]+"OTHERS")

    # Convert dicom2png
    dicom2png(input_Dicom2PNG,output_Dicom2PNG,str_parse)
    