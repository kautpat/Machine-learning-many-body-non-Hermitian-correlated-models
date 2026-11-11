# **Machine-learning-many-body-non-Hermitian-correlated-models**

This project investigates the 16-site 1D non-Hermitian AAH model (non-Hermitian due to imaginary potential modulation) under the presence of electronic correlations. The correlations lead to a many-body state whose quantum complexity can be quantified using correlation entropy. The work discovers a relationship between spikes in correlation entropy value and exceptional points (a characteristic non-hermitian feature), and trains a supervised neural network that calculates the correlation entropy from a reduced set of inputs. This reduction allows for predicting correlation entropy values across different system sizes, which is otherwise unfeasible as it grows as $O(N^2)$ where N is the number of sites. This reduction is useful as the correlation entropy can indicate the presence of exceptional points.

The masters thesis can be found [here](https://aaltodoc.aalto.fi/items/5e0f00c9-c519-4d8e-a5b0-cf6d1c0ab058).

***
 
## **Repository Structure**

```
thesis-project/
├── README.md
├── ML/           # ML code (entry point: neural_network.py)
├── Simulation code/
│   ├── non_interacting        # Baseline simulation without interactions
│   └── interacting            # Simulation including electronic interactions 
└── images/                         # Figures used in thesis
```

***
## **Simulation code**
Code of the non-Hermitian AAH model in intaeracting and non-interacting cases. The primary paramter that is varied across the thesis work is `v_I`, the imaginary component of the potential modulation, as there exists a phase transition through the point `v_I=2`.

### <u>Non-Interacting case</u>:
A skeleton code for the single particle non-Hermitian AAH model, which are  used for  calculations of exceptional points. Sytems with large number of  sites (~300) can be calculated using the single particle model as the matrices remain small, but special attention to 12 and 16 site models is given as they are among the largest possible in the interacting case. 

### <u>Interacting case</u>:
A skeleton code for the many body non-Hermitian AAH model, with functions for calculating correlation entropy using [joselado/dmrgpy](https://github.com/joselado/dmrgpy) library. Due to technical limitations the number of sites simulated in the models are 12 an 16. The code is built with HPC parallelization (triton) in mind and can be modified as required. Since we are interested in smooth evolution of `v_I` it remains as a for loop to pass arguments back into the dmrgpy methods.


***

## **ML code**

### <u>Data generation</u>:
Code to generate a large number of points for training the model, 10000+ data points were generated, a compiling script is included.

### <u>16_site-density-matrix</u>:
The neural network used in for the thesis, can be used directly. Uses several preprocessing steps such as PCA, normalisation, and data filtering. The model is made up of Dense and guassian layers with dropout, ReLU, and L2 regularization.

### <u>16-phase-data+model</u>:
Folder with test data of 16 site AAH model using the neural network for making visual comparisons.


***

## **Modifications to dmrgpy**

Based on [joselado/dmrgpy](https://github.com/joselado/dmrgpy) at commit `8eb44ce`.
Full diff: [compare view](https://github.com/kautpat/dmrgpy/compare/8eb44ce...thesis-version)

Main changes (all in `src/`):
- **[Wave function evolution]**: e.g. added support for smooth wavefunction evolution by adding additional arguments and modifying the State class (`def lowest_states` in `algebra.py`, `def get_gs_array` in `edchain.py`) 
- **[Exceptional points]**: e.g. added functions to aid in determining the presence of exceptional points (`def EP_checker` in `algebra.py`, `def get_ep` in `edchain.py`, `def get_ep` in `fermionchain.py`)

The code should work with a clone of `thesis-version` branch from [this link](https://github.com/kautpat/dmrgpy)