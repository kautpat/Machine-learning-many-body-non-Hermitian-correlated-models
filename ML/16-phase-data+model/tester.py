from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
import os
import joblib
from itertools import combinations


modelpath = '/scratch/work/patelk1/thesis_code/training_data/16_mf_ddd_10/'

def preprocess_new_data(X, scaler_path=modelpath+"scaler.pkl", pca_path=modelpath+"pca.pkl"):
    # Load fitted preprocessing objects
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)

    # Apply scaler + PCA
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    return X_pca

u_num = 80 # either no of U points OR no of v_R points
vI_num = 60
arr = np.zeros([u_num,vI_num])
# build 2d array

for i in range(u_num):
    Sc = np.load('/scratch/work/patelk1/thesis_code/training_data/phase_plot_data/data/data_{}/S_array1.npy'.format(i))
    arr[i,:] = Sc

# contour plots
x = np.linspace(0, 3, arr.shape[1]) 
y = np.linspace(0.2, 1.0, arr.shape[0])  
X, Y = np.meshgrid(x, y)

three_color_pastel = LinearSegmentedColormap.from_list(
    "three_color_pastel",
    ["#add8e6", "#ffffff", "#fbb1bd"]
)
vmin = 0.0
vmax = arr.max()

plt.figure(figsize=(8, 6))
cf = plt.contourf(X, Y, arr, levels=300, cmap='viridis')#three_color_pastel)#
c = plt.contour(X, Y, arr, levels=50, colors='black', linewidths=0.5)
plt.colorbar(cf, label='Sc value')

plt.xlabel(r'$v_I/t$')
plt.ylabel(r'Hubbard U')
plt.title(r'Phase diagram: $v_R=0$, $\phi = 0$, $\mu=0$, $t_2=0$')#2\pi/5
plt.tight_layout()
plt.savefig('true-phase_fixed_cmap.png')


def get_correlators(chain_size,correlation_matrix,density_density_matrix,number_of_sites):
    ccc = np.array([],dtype=complex)
    ddd = np.array([],dtype=complex)
    d2  = np.array([],dtype=complex)
    for ii in range(int(chain_size/2)-2,int(chain_size/2)+2):
        #ccc = np.append(ccc,correlation_matrix[ii,ii-4:ii])         # takes 2 sites before diag
        #ccc = np.append(ccc,correlation_matrix[ii,ii+1:ii+5])       # takes 2 sites after diag element

        ddd = np.append(ddd,density_density_matrix[ii,ii-4:ii])     # takes 2 sites before diag
        ddd = np.append(ddd,density_density_matrix[ii,ii+1:ii+5])   # takes 4 sites after diag element
        #d2 = np.append(d2,np.array([a - b for a, b in combinations(density_density_matrix[ii,ii+1:ii+5], 2)]))
    return np.concatenate((ccc.real, ddd.real),axis=None)                      # ccc.real, ccc.imag,

full_phase = np.zeros([u_num,vI_num])
model = tf.keras.models.load_model(modelpath+"my_model")



#target_scaler = joblib.load(modelpath+"target_scaler.pkl")
y_max = 0.10332511266568514#0.15043996876419904 #0.16378407313131096 #0.11854733175580821 #0.27688321460618603#0.15235948742729444 #0.10244077220700766
y_min = 0
for i in range(u_num):
    path = "/scratch/work/patelk1/thesis_code/training_data/phase_plot_data/data/data_{}".format(i)

    correlation_matrix = np.load(path + "/C_matrix1.npy")
    density_density_matrix = np.load(path + "/F_matrix1.npy")
    sites=4
    
    for j in range(vI_num):
        print(correlation_matrix[j].shape)
        phase_input = get_correlators(16,(correlation_matrix[j]), (density_density_matrix[j]), sites)
        #full_phase[i,j] = model.predict(phase_input.reshape(1, phase_input.shape[0]))*y_max
        pred = model.predict(preprocess_new_data(phase_input.reshape(1, -1)))#.reshape(-1,phase_input.shape[0],1))
        #pred = model.predict(phase_input.reshape(-1,6,6,2))
        full_phase[i,j] = pred.item() * (y_max - y_min) + y_min
        #full_phase[i,j] = target_scaler.inverse_transform(pred.reshape(-1,1))



'''
plt.figure(figsize=(8, 6))
cf = plt.contourf(X, Y, full_phase, levels=300, cmap='viridis')#three_color_pastel)#
c = plt.contour(X, Y, full_phase, levels=50, colors='black', linewidths=0.5)
plt.colorbar(cf, label='Sc value')

plt.xlabel(r'$v_I/t$')
plt.ylabel(r'Hubbard U')
plt.title(r'Predicted phase diagram: $v_R=0$, $\phi = 0$, $\mu=0$, $t_2=0$')
plt.tight_layout()
#plt.savefig('pred-phase-16_mf_ddd_10.png')


plt.figure(figsize=(8, 6))
cf = plt.contourf(X, Y, abs(arr-full_phase), levels=300, cmap='viridis')#,vmin=vmin,vmax=vmax)#three_color_pastel)#
#c = plt.contour(X, Y, full_phase, levels=50, colors='black', linewidths=0.5)
plt.colorbar(cf, label='Sc value')
plt.xlabel(r'$v_I/t$')
plt.ylabel(r'Hubbard U')
plt.title(r'real - pred')
plt.tight_layout()
plt.savefig('diff.png')
'''

def fidelity(true,pred):
    numer =  np.abs( np.mean(pred*true) - np.mean(pred) * np.mean(true) )
    denom = np.sqrt( (np.mean(true*true) - np.mean(true)*np.mean(true)) * (np.mean(pred*pred) - np.mean(pred)*np.mean(pred)) )
    f = numer / denom
    return f

print(fidelity(arr,full_phase))

plt.figure(dpi=300)
plt.scatter(arr.flatten(),full_phase.flatten())
plt.xlabel(r'Target $S_C$')
plt.ylabel(r'Predicted $S_C$')
plt.plot(np.linspace(0,np.max(arr.flatten()),20),np.linspace(0,np.max(arr.flatten()),20),'r')
plt.savefig('test-vs-predict.png')
plt.close()