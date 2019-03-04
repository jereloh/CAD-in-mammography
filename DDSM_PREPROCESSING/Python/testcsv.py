'''
readCSV
[Goal]: Read file into an matrix to correlate the folder names to the correct status of the images
'''
import pandas

def readCSV():
    data_path = "/Users/xfler/Documents/GitHub/Year4_FYP/Images/RAW_CBIS_DDSM/mass_case_description_train_set.csv"
    #read csv
    corr_df =  pandas.read_csv(data_path)
    
    print (corr_df)

readCSV()