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

    # Counter for progress bar
    i = 1

    # Iterate through list of pngs retrieved
    for file in list_of_PNGs:
        try:               
            
            img = Image.open(file) # image extension *.png,*.jpg
            new_width  = width
            new_height = height
            img = img.resize((new_width, new_height), Image.ANTIALIAS) # High quality downsampling filter
            
            if (folder_Name == "AlexNet"):
                # To RGB (ALEXNET NEEDS THIS)
                img_RGB = Image.new("RGB", img.size)
                img_RGB.paste(img)
                img = img_RGB
            
            # Obtain file name
            png_name = file.split("/")

            # Save converted Image
            img.save(os.path.join(img_out,png_name[len(png_name)-1]))
            
            # Progress bar
            print ("[",i,"/",len(list_of_PNGs),"] :",png_name[len(png_name)-1])
            i += 1
        except:
            print("failed")
            quit()

    # Double check if images are correct size
    list_of_Converted_PNGs = glob.glob( img_out+'/*.png', recursive=True)
    j = 1
    cvt_flag = True
    for cvt_file in list_of_Converted_PNGs:
        cvt_img = Image.open(cvt_file)
        cvt_width, cvt_height = cvt_img.size
        if (cvt_width != width and cvt_height != height):
            cvt_flag = False
            print("Wrong width and height in:",cvt_file)
        j+=1
    if (cvt_flag == True):
        print ("Check OK")        

'''
[MAIN]
'''
downsample_Alexnet("AlexNet",272,272)