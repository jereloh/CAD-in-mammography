from PIL import Image
import os, argparse, glob
import time  
import concurrent.futures
from tqdm import tqdm
import multiprocessing

# [Function] Create necessary folder function
def chkFolder(flder_path):
    if not os.path.exists(flder_path):
        print ("Folder don't exist. Making..")
        os.makedirs(flder_path)

# [Function] Create downsampling images to size
def downsample(file):
    size = args["size"]
    width,height = size.split("_")

    img_out = os.path.join(args["input"],args["name"]+"_"+args["size"])
    try:               
        img = Image.open(file) # image extension *.png,*.jpg
        new_width  = int(width)
        new_height = int(height)
        img = img.resize((new_width, new_height), Image.ANTIALIAS) # High quality downsampling filter
        # Obtain file name

        if os.name == 'posix':
            png_name = file.split("/")
        elif os.name == 'nt':
            png_name = file.split("\\")
        else:
            print ("Operating System not supported: "+os.name)
            quit()
        img.save(os.path.join(img_out,png_name[len(png_name)-1]))
    except:
        print("failed")
        quit()        

# Construct argument parsing
argparser = argparse.ArgumentParser()
argparser.add_argument( "-n", "--name", required=True,
	help="Please insert name of folder to be generated.")
argparser.add_argument( "-s", "--size", required=True,
	help="Please insert width size to downsample e.g. 272_272, WIDTH_HEIGHT.")
argparser.add_argument( "-i", "--input", required=True,
	help="Please insert input folder path.")
args = vars(argparser.parse_args())

size = args["size"]
try:
    width,height = size.split("_")
except:
    print("wrong input for size")

img_out = os.path.join(args["input"],args["name"]+"_"+args["size"])
chkFolder(img_out)

list_of_PNGs = glob.glob( os.path.join(args["input"],'*.png'), recursive=True) # Ensures only look for .png

# An attempt at parallel processing
with concurrent.futures.ThreadPoolExecutor(multiprocessing.cpu_count()) as executor: 
    list(tqdm(executor.map(downsample, list_of_PNGs), total=len(list_of_PNGs)))
