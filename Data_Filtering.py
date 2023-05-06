import numpy as np
import pandas as pd
import os
from fnmatch import fnmatch


data = pd.read_csv('./data/DFBS_extracted.csv', index_col='Unnamed: 0')
print(data.head())


all_tiff_files = []
listOfFiles = os.listdir('./data/images/')
pattern = "*.tiff"
for entry in listOfFiles:
    if fnmatch(entry, pattern):
        all_tiff_files.append('./data/images/'+entry)

all_tiff_files[0], len(all_tiff_files)


from time import perf_counter


new_index = 0
arr_data = []
t1 = perf_counter()
for index, row in data.iterrows():
    # if index == 0: continue
    glon = row["_Glon"]
    glat = row["_Glat"]
    raj = row["_RAJ2000"]
    dej = row["_DEJ2000"]
    cl = row["Cl"]
    name = row["Name"]
    vmag = row["Vmag"]
    z = row["z"]
    plate = row["plate"]
    dx = row["dx"]
    dy = row["dy"]
    # if cl == "C" or cl == "PN" or cl == "cv" : 
        # continue
        
    for i in range(len(all_tiff_files)):
        tiff_index, file_name = all_tiff_files[i].split("/")[-1].split(".tiff")[0].split('__');
        if (name == file_name) and (index == int(tiff_index)):
            arr_data.append([name, cl, all_tiff_files[i]])
            break

print(perf_counter() - t1)
