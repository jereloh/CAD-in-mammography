from PIL import Image
import os, argparse, glob

# Construct argument parsing
argparser = argparse.ArgumentParser()
argparser.add_argument( "-i", "--input", required=True,
	help="Please insert input folder path.")
args = vars(argparser.parse_args())

# [Function] Create necessary folder function
def chkFolder(flder_path):
    if not os.path.exists(flder_path):
        print ("Folder don't exist. Making..")
        os.makedirs(flder_path)

# [Function] Create downsampling images to size
def downsample_Alexnet(folder_Name, width, height):
    img_out = os.path.join(args["input"],folder_Name+"_"+str(width) +"by"+ str(height))
    chkFolder(img_out)

    list_of_PNGs = glob.glob( args["input"]+'/*.png', recursive=True) # Ensures only look for .png

    # Counter
    i = 0
    for file in list_of_PNGs:
        try:               
            print (file)
            img = Image.open(file) # image extension *.png,*.jpg
            new_width  = width
            new_height = height
            img = img.resize((new_width, new_height), Image.ANTIALIAS) # High quality downsampling filter
            # Obtain file name
            png_name = file.split("/")
            img.save(os.path.join(img_out,png_name[len(png_name)-1]))
        except:
            print("failed")
            quit()
        
        # Progress bar
            print ("[",i,"/",len(list_of_PNGs),"]", end='')
            i += 1

'''
[MAIN]
'''
downsample_Alexnet("AlexNet",272,272)