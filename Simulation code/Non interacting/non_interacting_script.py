import numpy as np
from matplotlib import pyplot as plt

def aah_hamiltonian(L, t=1.0, v_R=1.0, alpha=(np.sqrt(5)-1)/2, phi=0.0,
                    t_im=0.0, v_I=0.0, periodic=False):
    """
    Construct a non-interacting tight-binding Hamiltonian for the AAH model.

    Parameters
    ----------
    L : int
        Number of lattice sites.
    t : float
        Real nearest-neighbor hopping amplitude.
    v_R : float
        Real onsite potential strength.
    alpha : float
        Incommensurate modulation frequency (irrational recommended).
    phi : float
        Phase offset of onsite potential.
    t_im : float
        Imaginary contribution to hopping (non-Hermitian part).
    v_I : float
        Imaginary contribution to onsite potential (non-Hermitian part).
    periodic : bool
        Whether to impose periodic boundary conditions.

    Returns
    -------
    H : np.ndarray (complex)
        L x L Hamiltonian matrix (complex-valued).
    """
    H = np.zeros((L, L), dtype=np.complex128)

    # onsite terms
    for j in range(L):
        onsite_real = v_R * np.sin(2 * np.pi * alpha * j + phi)
        onsite_imag = v_I * np.sin(2 * np.pi * alpha * j + phi)

        H[j, j] = onsite_real + 1j * onsite_imag

    # hopping terms
    for j in range(L - 1):
        hopping_real = -t
        hopping_imag = -t_im
        H[j, j+1] = hopping_real + 1j*hopping_imag
        H[j+1, j] = hopping_real - 1j*hopping_imag  # Hermitian if t_im=0

    # periodic BC
    if periodic and L > 2:
        hopping_real = -t
        hopping_imag = -t_im
        H[0, L-1] = hopping_real + 1j*hopping_imag
        H[L-1, 0] = hopping_real - 1j*hopping_imag

    return H

def solve(M):
    values,vectors = np.linalg.eig(M)
    return sorteigen(values,vectors)

def eig_vector_matrix(v):
    """Makes innerproduct matrix from the given vectors"""
    m = v@(np.conjugate(v.T))
    return m

def sorteigen(eig,vs):
    """Return sorted eigenvalues and eigenvectors"""
    w = eig - np.min(eig.real) # smallest real part
    imweight = 0#1e-3 # how much to weight im when sorting 
    wz = w.real + imweight*w.imag
    vs = [y for (x,y) in sorted(zip(wz,vs),key=lambda x: x[0])]
    eig = [y.copy() for (x,y) in sorted(zip(wz,eig),key=lambda x: x[0])]
    return np.array(eig),vs


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def birothog_ordered(L_v, R_v, L_e, R_e):
    ord_1 = np.argsort(L_e.real*1000 - L_e.imag)
    ord_2 = np.argsort(R_e.real*1000 + R_e.imag)


    
    for i in range(L_v.shape[1]):
        norm = np.vdot(L_v[:, i], R_v[:, i])
        L_v[:, i] /= np.sqrt(norm)
        R_v[:, i] /= np.sqrt(norm)

    return L_v.conj().T @ R_v   

L = 144
t = 1
v_I = np.arange(0,5,0.01)
v_R = 0
phi = 0 #2*np.pi/6
t_im = 0
alpha = 89/144 #(np.sqrt(5)-1)/2 # 233/377#
det = []
det_bi = []
lamda = np.arange(0,2,0.001)
for i in range(v_I.shape[0]):
    H = aah_hamiltonian( L=L, t=t, v_R=v_R, alpha=alpha, phi=phi, t_im=t_im, v_I=v_I[i], periodic=True ) #np.array([[1j*lamda[i],1],[1,-1j*lamda[i]]],dtype=complex)
    R_e, R_v = solve(H) #right
    M = eig_vector_matrix(np.array(R_v))
    det.append(np.linalg.det(M))
    print(f'{(i+1)/v_I.shape[0]*100:2f}%\n')




print(np.array(det))
plt.figure(dpi=300)
plt.semilogy(v_I,abs(np.array(det)))
plt.xlabel(r'$v_I/t$')
plt.ylabel(r'Determinant')
plt.savefig('test_small_vr_nonni.png')