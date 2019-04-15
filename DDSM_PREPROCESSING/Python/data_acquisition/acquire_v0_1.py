'''
[Description]
Simple py script to sort and convert dicom to png from the CBIS_DDSM

[Goal] 
<~~> are arbitary values used for explanation of this script.
1. Read .csv file provided, to be placed within the input folder ./<name of folder>/
2. Read folder name which correlates to patient from .csv
3. Creates folders benign, malignant, normal within input folder ./<name of folder/BENIGN/,./<name of folder/MALIGNANT/ etc.
4. Obtains .dcm from ./<name of folder>/CBIS-DDSM/Mass-Training_<P_00001_LEFT_CC>/*/*/*.dcm file path
5. Converts .dcm into .png using <P_00001_LEFT_CC>.png as filename
6. Continues until all files within ./<name of folder>/CBIS-DDSM/ is completed
7. Check that based on .csv, all data have been converted (no. of conversions)

[Limitations]
1. Script's recursive file search uses glob expects only a single file .dcm to be found
2. Script is written for unix/linux systems only
3. Script expects .csv to have these headers patient_id,left or right breast,image view,pathology to determine filename
4. Script treats pathology data BENIGN_WITHOUT_CALLBACK as the same as BENIGN 
5. Script only catered for pathology values BENIGN, MALIGNANT, all other values are grouped as OTHERS
'''
# Call the necessary imports
import os, argparse, glob
import pydicom, png
import numpy, pandas

# construct argument parsing
argparser = argparse.ArgumentParser()
argparser.add_argument( "-i", "--input", required=True,
	help="Please insert input folder path.")
argparser.add_argument( "-csv", required=True,
	help="Please insert csv filename. E.g. -csv mass_case_description_train_set.csv")
args = vars(argparser.parse_args())

# Referenced from https://github.com/pydicom/pydicom/issues/352
def dicom2png(src, out_folder, file_name): 
    try:
        ds = pydicom.dcmread(src)
        shape = ds.pixel_array.shape

        # Convert to float to avoid overflow or underflow losses.
        image_2d = ds.pixel_array.astype(float)

        # Rescaling grey scale between 0-255 (image in grey scale anyhows)
        image_2d_scaled = (numpy.maximum(image_2d,0) / image_2d.max()) * 255.0

        # Convert to uint
        image_2d_scaled = numpy.uint8(image_2d_scaled)

        # Writes .PNG file
        with open(os.path.join(out_folder,file_name)+'.png' , 'wb') as png_file:
            w = png.Writer(shape[1], shape[0], greyscale=True)
            w.write(png_file, image_2d_scaled)
        print(file_name+".png","converted")
    except:
        print('[ERROR] Could not convert:',src)
        quit()

# [Function] readCSV
def readCSV(data_path):
    try:
        corr_df =  pandas.read_csv(data_path)
        # Select only required ids and put into dataframe
        print ("Reading:",data_path)
        return corr_df[['patient_id','left or right breast','image view','pathology']]
    except:
        print ('[ERROR] Could not read csv:',data_path)
        quit()

# [Function] Create necessary folder function
def chkFolder(flder_path):
    if not os.path.exists(flder_path):
        print ("Folder don't exist. Making..")
        os.makedirs(flder_path)

# [Function] Subfolder finder using glob
def dcmFullPath(str_parse):
    # Find all subfolders using glob module, iterate through and do sth based on file path name
    files = glob.glob( args["input"]+"*/*"+str_parse+"/" + '/*/*/*.dcm', recursive=True)
    if (len(files) == 1):
        return files[0]
    else:
        print ("[ERROR] More than 1 file found", files)
        quit()

# --- "MAIN" ---

# Read .CSV, put into dataframe
corr_df_datapath = args["input"]+ args["csv"]
corr_df = readCSV(corr_df_datapath)

# Check folders Malignant and Benign created
chkFolder(args["input"]+"BENIGN")
chkFolder(args["input"]+"MALIGNANT")
chkFolder(args["input"]+"OTHERS") # In Case there are other values apart from BENIGN and MALIGNANT

print("Converting...")

# Counter i, for checking number of conversions
i = 1

# Iterate through dataframe 
for index, row in corr_df.iterrows():

    # Place the parse string together to search the file path for it.
    str_parse = row["patient_id"]+"_"+row["left or right breast"]+"_"+row["image view"] 

    # Generate file path of .dcm
    input_Dicom2PNG = dcmFullPath(str_parse)

    # Sort input into relevant folders
    if "BENIGN" in row["pathology"] :  
        output_Dicom2PNG = (args["input"]+"BENIGN")      
    elif "MALIGNANT" in row["pathology"] :      
        output_Dicom2PNG = (args["input"]+"MALIGNANT")
    else:
        output_Dicom2PNG = (args["input"]+"OTHERS")

    # Progress bar
    print ("[",i,"/",len(corr_df),"]", end='')

    # Convert dicom2png
    dicom2png(input_Dicom2PNG,output_Dicom2PNG,str_parse)
    if i!= len(corr_df):
        i += 1

if i == len(corr_df):
    print("All",i,"data found in",args["csv"],"have been converted.")
else:
    print("Not all data found in",args["csv"],"have been converted.",i,"converted. .csv has",len(corr_df),"rows.")