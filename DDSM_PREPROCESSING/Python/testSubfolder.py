import glob

files = glob.glob('/Users/xfler/Documents/GitHub/Year4_FYP/Images/RAW_CBIS_DDSM/Mass-Training/CBIS-DDSM/Mass-Training_P_00001_LEFT_CC/**/**/*.dcm', recursive=True)
print (files)