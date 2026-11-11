# 1. libraries
import sys
sys.path.append("/scratch/work/patelk1/dmrgpy/src")
from dmrgpy import fermionchain
import numpy as np
import cmath as cm
from numpy import linalg
from numpy.random import seed
import random
import itertools
from matplotlib import pyplot as plt

import time
import os

# 2. Functions
def get_fc(n,t,t2,v_I,v_R,alpha,U,mu,phi):
    '''This function returns the many-body Hamiltonian'''
    fc = fermionchain.Fermionic_Chain(n)
    
    H = 0
    
    for i in range(n-1):
        H = H + t*fc.Cdag[i]*fc.C[i+1]              # Hopping terms
 
    for i in range(n-2):
        H = H + t2*fc.Cdag[i]*fc.C[i+2]   

    H = H + H.get_dagger() # Make Hermitian

    for i in range(n-1):
        H = H + U*(fc.N[i+1]-0.5)*(fc.N[i]-0.5)
    # Diagonal terms 
    for i in range(n):
        H = H + mu*fc.Cdag[i]*fc.C[i]
    
    # Make non-Hermitian
    for i in range(n):
        H = H + (v_R + 1j*v_I) * np.sin(2 * np.pi * alpha * i + phi)*fc.N[i]

    fc.set_hamiltonian(H) # initialize the Hamiltonian
    
    fc.maxm = 120
    fc.nsweeps = 120   
    return fc

def one_particle_correlation_entropy(n,t,t2,v_I,v_R,alpha,U,mu,phi):

    def correlation_entropy(fc,wf):
        corr_matrix=np.zeros([n,n],dtype = complex)
        for ii in range(n):
            for jj in range(n):
                corr=wf.aMb(fc.Cdag[ii]*(fc.C[jj]),wf) 
                corr_matrix[ii][jj]=corr 
        las2 = linalg.eigvals(corr_matrix) 
        Sc0=0
        for k in range(0,len(las2)):
            if las2[k]==0:
                Sc0+=0
            else:
                Sc0+=-las2[k]*cm.log(las2[k])
        print(Sc0)
        return Sc0.real/n, corr_matrix
    
    def correlation4_matrix_full(fc,wf):
        """This function returns four-point correlation matrix"""
        corr_matrix=np.zeros((n,n),dtype=complex)
        for ii in range(0,n):
            for jj in range(0,n):
                corr_matrix[ii,jj] = wf.aMb((fc.N[ii])*(fc.N[jj]),wf) - (wf.aMb((fc.N[ii]),wf)) * (wf.aMb((fc.N[jj]),wf))
        return corr_matrix

    fc = get_fc(n=n, t=t, t2=t2, v_I=v_I, v_R=v_R, alpha=alpha, U=U, mu=mu, phi=phi)
    wf = fc.get_gs(mode="ED")
    Sc,corr_m = correlation_entropy(fc,wf)
    corr_m4 = correlation4_matrix_full(fc,wf)

    return Sc,corr_m,corr_m4


# 3. Parameters and containers
# 3.1 Basic parameters
n = 16  # number of sites of the Hubbard model
t = 1   # nearest-neighbour hopping, also the unit energy of the system

# 3.2 Containers for correlation entropy
Sc_list=[] # to be used for appending correlation entropy

C_matrix_list=[]
F_matrix_list=[]



dtype = [
    ('t2', 'f4'),      # hopping
    ('mu', 'f4'),      # chemical potential
    ('U','f4'),        # Hubbard U
    ('vR', 'f4'),      # quasi-periodic potential real part
    ('vI', 'f4'),      # quasi-periodic potential imaginary part
    ('alpha', 'f4'),   # irrational num usually for making quasi periodic.
    ('phi', 'f4'),     # phase
    ('seed','i8'),     # seed of rng data
]

# 4. Obtaining data
# 4. Loop
NN=50 # Number of examples
init_data = np.zeros(NN, dtype=dtype)

job_id = int(os.environ.get("SLURM_JOB_ID", 0))
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
seed = job_id * 1000 + task_id 
print(seed)
rng = np.random.default_rng(seed=seed)
zz=0
while len(Sc_list) < NN:
    print(zz)
    # Generating random variables
    t2      = t*rng.random()
    mu      = t*rng.random()
    U       = 2*t*rng.random()
    alpha   = rng.random()#(np.sqrt(5)-1)/2.0    #should this be irrational?
    vR      = 3*t*rng.random()
    vI      = 3*t*rng.random()
    phi     = 2*np.pi*rng.random()

    init_data[zz] = (t2,mu,U,vR,vI,alpha,phi,seed)

    Sc, corr_mat, corr_mat4 = one_particle_correlation_entropy(n=n, t=t, t2=t2, v_I=vI, v_R=vR, alpha=alpha, U=U, mu=mu, phi=phi)
    # Add code to generate particle-particle correlation matrix, density-density matrix and correlation entropy form particle-particle correlation matrix
    Sc_list.append(Sc)
    C_matrix_list.append(corr_mat)
    F_matrix_list.append(corr_mat4)
    zz+=1
    

# 5. Saving data
C_matrix=np.array(C_matrix_list)
F_matrix=np.array(F_matrix_list)
S_array=np.array(Sc_list)

#np.save("data.npy", data_array)
np.save("data_array1.npy", init_data)
np.save("C_matrix1.npy", C_matrix)
np.save("F_matrix1.npy", F_matrix)
np.save("S_array1.npy", S_array)