from os import walk
import os
import numpy as np

f = []
C_matrix_list=[]
F_matrix_list=[]
data_array_list=[]
S_array_list=[]
for (dirpath, dirnames, filenames) in walk(os.getcwd()):
    os.chdir(dirpath) 

    print(dirpath)
    for f0 in filenames: 
        if "C_matrix1.npy" in f0: 
            try:
                m = np.load(f0) # load
                C_matrix_list.append(m)
                print("Converting ",f0)
            except:
                print("Something wrong with ",f0)
    
        if "F_matrix1.npy" in f0: 
            try:
                m = np.load(f0) # load
                F_matrix_list.append(m)
                print("Converting ",f0)
            except:
                print("Something wrong with ",f0)

        if "data_array1.npy" in f0: 
            try:
                m = np.load(f0) # load
                data_array_list.append(m)
                print("Converting ",f0)
            except:
                print("Something wrong with ",f0)

        if "S_array1.npy" in f0: 
            try:
                m = np.load(f0) # load
                S_array_list.append(m)
                print("Converting ",f0)
            except:
                print("Something wrong with ",f0)

for ii in range(len(C_matrix_list)):
    if ii==0:
        nC_matrix=C_matrix_list[ii]
    else:
        nC_matrix=np.concatenate((nC_matrix,C_matrix_list[ii]), axis=0)

for ii in range(len(F_matrix_list)):
    if ii==0:
        nF_matrix=F_matrix_list[ii]
    else:
        nF_matrix=np.concatenate((nF_matrix,F_matrix_list[ii]), axis=0)

for ii in range(len(data_array_list)):
    if ii==0:
        ndata_array=data_array_list[ii]
    else:
        ndata_array=np.concatenate((ndata_array,data_array_list[ii]), axis=0)

for ii in range(len(S_array_list)):
    if ii==0:
        nS_array=S_array_list[ii]
    else:
        nS_array=np.concatenate((nS_array,S_array_list[ii]), axis=0)

print(nC_matrix.shape)
print(nF_matrix.shape)
print(ndata_array.shape)
print(nS_array.shape)

np.save("/filepath/data_16_4/data_array.npy", ndata_array)
np.save("/filepath/data_16_4/C_matrix.npy", nC_matrix)
np.save("/filepath/data_16_4/F_matrix.npy", nF_matrix)
np.save("/filepath/data_16_4/S_array.npy", nS_array)