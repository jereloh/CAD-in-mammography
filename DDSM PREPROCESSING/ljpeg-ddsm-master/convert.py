import os

path_to_ddsm = "/Users/xfler/Documents/GitHub/Year4_FYP/DDSM_PREPROCESSING/ljpeg-ddsm-master/case4000/"

for root, subFolders, file_names in os.walk(path_to_ddsm):
    for file_name in file_names:
        if ".LJPEG" in file_name:
            ljpeg_path = os.path.join(root, file_name)
            out_path = os.path.join(root, file_name)
            out_path = out_path.split('.LJPEG')[0] + ".jpg"
            print("Converting")
            cmd = './ljpeg.py "{0}" "{1}" --visual --scale 1.0'.format(ljpeg_path, out_path)
            os.system(cmd)
        else:
            print("failed")

print('done')