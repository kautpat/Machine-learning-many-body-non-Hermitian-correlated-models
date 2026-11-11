"""
Created for obtaining two-point correlators, four-point correlators and related correlation entropy on a n site Hubbard model

@author: Faluke Aikebaier
"""

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
from numpy.random import seed
import random
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
        H = H + U*(fc.N[i+1]-0.5)*(fc.N[i]-0.5)     # Hubbard interaction term

    for i in range(n-2):
        H = H + t2*fc.Cdag[i]*fc.C[i+2]   

    H = H + H.get_dagger() # Make Hermitian
        
    # Diagonal terms 
    for i in range(n):
        H = H + mu*fc.Cdag[i]*fc.C[i]
    
    # Make non-Hermitian
    for i in range(n):
        H = H + (v_R + 1j*v_I) * np.sin(2 * np.pi * alpha * i + phi)*fc.N[i]

    fc.set_hamiltonian(H) # initialize the Hamiltonian
    
    fc.maxm = 50
    fc.nsweeps = 50   
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
if __name__ == "__main__":
    num1 = sys.argv[1]
    print(num1)

    n = 16# number of sites of the Hubbard model
    t = 1 # nearest-neighbour hopping, also the unit energy of the system
    alpha = (np.sqrt(5)-1)/2.0
    v_I = np.arange(0,3,0.05)
    np.save("./corr_dat/v_I.npy",v_I)
    v_R = 0
    U = np.arange(0.1,0.5,0.005)[int(num1)]
    l = v_I.shape[0]
    t2 = 0
    mu = 0
    phi = 0
    # 3.2 Containers for correlation entropy
    Sc_list=[] # to be used for appending correlation entropy

    C_matrix_list=[]
    F_matrix_list=[]

    for i in range(l):
        Sc, corr_mat, corr_mat4 = one_particle_correlation_entropy(n=n, t=t, t2=t2, v_I=v_I[i], v_R=v_R, alpha=alpha, U=U, mu=mu, phi=phi)

        Sc_list.append(Sc)
        C_matrix_list.append(corr_mat)
        F_matrix_list.append(corr_mat4)

    # 5. Saving data
    C_matrix=np.array(C_matrix_list)
    F_matrix=np.array(F_matrix_list)
    S_array=np.array(Sc_list)

    np.save("C_matrix1.npy", C_matrix)
    np.save("F_matrix1.npy", F_matrix)
    np.save("S_array1.npy", S_array)
