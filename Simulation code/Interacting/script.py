# 1. libraries
import sys
sys.path.append("/FilePath/dmrgpy/src") # Add file path of drmrgpy source code
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
from scipy import stats

import time
import os

# 2. Global Variable
prev_wf = None   # used for maintaing a record of previous wavefunction to enable smooth wavefuntion evolution
prev_e  = None   # used for maintaing a record of previous groundstate energy to enable smooth wavefuntion evolution

# 3. Functions
def get_fc(n,t,t2,v_I,v_R,alpha,U,mu,phi):
    '''This function returns the many-body Hamiltonian'''
    fc = fermionchain.Fermionic_Chain(n)
    
    H = 0
    
    for i in range(n-1):
        H = H +  t*fc.Cdag[(i)%n]*fc.C[(i+1)%n]              # Hopping terms
    for i in range(n-2):
        H = H + t2*fc.Cdag[(i)%n]*fc.C[(i+2)%n]             # Second Hopping

    H = H + H.get_dagger() # Make Hermitian

    for i in range(n-1):
        H = H + U*(fc.N[(i+1)%n]-0.5)*(fc.N[(i)%n]-0.5)                             # Hubbard interaction term

    for i in range(n):    
        H = H + mu*fc.Cdag[i]*fc.C[i]                                           # Chemical potential
        H = H + (v_R + 1j*v_I) * np.sin(2 * np.pi * alpha * i + phi)*fc.N[i]    # Onsite modulation

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

    global prev_wf
    global prev_e
    fc = get_fc(n=n, t=t, t2=t2, v_I=v_I, v_R=v_R, alpha=alpha, U=U, mu=mu, phi=phi)
    wf = fc.get_gs(mode="ED", psi_prev=prev_wf, e_prev=prev_e)
    prev_wf = wf
    prev_e  = wf.e
    sc,crm = correlation_entropy(fc,wf)

    det_val = fc.get_ep(n=5)
    
    return sc, crm, det_val



# 4. Parameters and containers
# 4.1 Basic parameters
if __name__ == "__main__":

    num1 = sys.argv[1]
    print(num1)
    n = 12      # number of sites
    t = 1       # nearest-neighbour hopping, also the unit energy of the system
    t2 = 0
    mu = 0
    phi = 0
    alpha = np.arange(0.12,0.9,0.03)[int(num1)] #27
    v_I = np.arange(0,5,0.05)
    v_R = 0
    U = 0.5
    print(n)
    Sc=[]
    det=[]
    for i in range(v_I.shape[0]):
        Sc0,_,det0 = one_particle_correlation_entropy(n=n, t=t, t2=t2, v_I=v_I[i], v_R=v_R, alpha=alpha, U= U, mu=mu, phi=phi)
        det.append(det0)
        Sc.append(Sc0)
    np.save('Sc.npy',np.array(Sc))
    np.save('det.npy',np.array(det))