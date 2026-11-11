# 1. Libraries
import numpy as np
import seaborn as sns
import pandas as pd
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks, optimizers
from autokeras import StructuredDataRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    median_absolute_error
)
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from itertools import combinations

# 1. Functions
def smape(y_true, y_pred, eps=1e-8):
    """Symmetric mean absolute percentage error (safe near zero)."""
    return 100 * np.mean(
        2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)
    )

def evaluate_regression(y_true, y_pred):
    """
    Compute multiple regression error metrics and print them.
    y_true, y_pred should be 1D numpy arrays.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae  = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # %
    smape_val = smape(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    # correlations
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)

    print(f"MAE       : {mae:.6f}")
    print(f"Median AE : {medae:.6f}")
    print(f"RMSE      : {rmse:.6f}")
    print(f"MAPE      : {mape:.3f} %")
    print(f"SMAPE     : {smape_val:.3f} %")
    print(f"R²        : {r2:.6f}")
    print(f"Pearson r : {pearson_corr:.6f}")
    print(f"Spearman  : {spearman_corr:.6f}")

    return {
        "MAE": mae,
        "MedianAE": medae,
        "RMSE": rmse,
        "MAPE": mape,
        "SMAPE": smape_val,
        "R2": r2,
        "Pearson": pearson_corr,
        "Spearman": spearman_corr,
    }


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
    return np.concatenate((ccc.real, ddd.real),axis=None)                    # ccc.real, ccc.imag,

def get_entrance(chain_size,correlation_matrices,density_density_matrices,number_of_sites,correlation_entropy):
    entrance_list=[]
    weight_list   = []
    bin1_c = 0  # 0->0.3
    bin2_c = 0
    bin3_c = 0
    bin4_c = 0
    np.random.seed(443)
    indices = np.arange(correlation_matrices.shape[0])
    np.random.shuffle(indices)
    for ii in indices:
        #if correlation_entropy[ii]>= 0 and correlation_entropy[ii] <= 0.001:
        #    bin4_c += 1
        #    if bin4_c>500:
        #        continue
        
        if correlation_entropy[ii]>= 0 and correlation_entropy[ii] <= 0.01:
            bin1_c += 1
            if bin1_c>4000:
                continue
        elif correlation_entropy[ii]> 0.01 and correlation_entropy[ii] <= 0.02:
            bin2_c += 1
            if bin2_c>4000:
                continue
        elif correlation_entropy[ii]> 0.02 and correlation_entropy[ii] <= 0.03:
            bin3_c += 1
            if bin3_c>4000:
                continue
        '''
        if correlation_entropy[ii] <= 0.04:
            weight = 1.0
        else:
            weight = 10.0
        '''
        
        weight = 1
        
        feature=get_correlators(chain_size,correlation_matrices[ii],density_density_matrices[ii],number_of_sites)
        target=correlation_entropy[ii]
        entrance_list.append(np.hstack([(feature),target]))
        weight_list.append(weight)

    print(bin1_c,bin2_c,bin3_c,bin4_c)
    return np.array(entrance_list),np.array(weight_list)

# 2. Data
path = "/scratch/work/patelk1/thesis_code/training_data/data_16"

correlation_matrix = np.load(path + "_4/C_matrix.npy")
density_density_matrix = np.load(path + "_4/F_matrix.npy")
correlation_entropy = np.load(path + "_4/S_array.npy")
print("loaded data")

'''
correlation_matrix = np.vstack([np.load(path + "/C_matrix.npy"),np.load(path + "_1/C_matrix.npy"),np.load(path + "_3/C_matrix.npy")])
density_density_matrix = np.vstack([np.load(path + "/F_matrix.npy"),np.load(path + "_1/F_matrix.npy"),np.load(path + "_3/F_matrix.npy")])
correlation_entropy = np.concatenate((np.load(path + "/S_array.npy"),np.load(path + "_1/S_array.npy"),np.load(path + "_3/S_array.npy")))
print("loaded data")
print(correlation_matrix.shape,density_density_matrix.shape,correlation_entropy.shape)
'''


print(max(correlation_entropy))
# -----------------------------------------------------------------------------------------------------------#

N = 16 #Size
sites=4
training_data,training_weights = get_entrance(N,(correlation_matrix), (density_density_matrix), sites, correlation_entropy)

data_shuffled, weights_shuffled = shuffle(training_data,training_weights, random_state=42)
data_features = (data_shuffled[:,0:training_data.shape[1]-1])
data_target   = abs(data_shuffled[:,training_data.shape[1]-1])
sample_weights = weights_shuffled

trainX, testX, trainY, testY=train_test_split(data_features, data_target, test_size=0.2, random_state=42)
y_min, y_max = trainY.min(), trainY.max()
trainY = (trainY - y_min) / (y_max - y_min)
testY  = (testY  - y_min) / (y_max - y_min)
print("Created train-test data")
print(y_min, y_max)
# -----------------------------------------------------------------------------------------------------------#
scaler = StandardScaler()
trainX_scaled = scaler.fit_transform(trainX)
testX_scaled  = scaler.transform(testX)
print(trainX_scaled.shape)

pca = PCA(n_components=0.99, svd_solver="full")
trainX_pca = pca.fit_transform(trainX_scaled)
testX_pca  = pca.transform(testX_scaled)

#trainX_pca = np.expand_dims(trainX_pca, axis=-1)  # shape: (N, F, 1)
#testX_pca  = np.expand_dims(testX_pca, axis=-1)
print(trainX_pca.shape)
print(testY.shape)
dump(scaler, "scaler.pkl")
dump(pca, "pca.pkl")
# -----------------------------------------------------------------------------------------------------------#

# 3. Model
dropout_rate = 0.2
weight_decay = 1e-5
#,kernel_regularizer=regularizers.l2(weight_decay)
model = tf.keras.Sequential([
    layers.GaussianNoise(0.01,input_shape=[trainX_pca.shape[1]]),
    layers.Dense(512, activation='relu',kernel_regularizer=regularizers.l2(weight_decay)),
    layers.Dropout(dropout_rate),
    layers.Dense(512, activation='relu',kernel_regularizer=regularizers.l2(weight_decay)),
    layers.Dropout(dropout_rate),
    layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(weight_decay)),
    layers.Dropout(dropout_rate),
    layers.Dense(units=1, activation='linear')])

def learning_rate_scheduler(epoch):
    initial_learning_rate = 0.001  # Initial learning rate
    if epoch <= 1:
        return initial_learning_rate
    else:
        decay = 0.975  # Learning rate decay factor
        lr = initial_learning_rate * (decay ** epoch)
        return lr
# Create a callback to adjust the learning rate during training
lr_scheduler_callback = keras.callbacks.LearningRateScheduler(learning_rate_scheduler)

model.compile(
    optimizer=tf.optimizers.Adam(),
    loss='log_cosh')#tf.keras.losses.Huber(delta=0.001))#'mean_absolute_error')

print("Just before train")


history = model.fit(
    trainX_pca,
    trainY,
    sample_weight=sample_weights[:len(trainX_pca)],
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.#,batch_size=1024,
    validation_split = 0.2,
    callbacks=[lr_scheduler_callback])



def fidelity(true,pred):
    numer =  np.abs( np.mean(pred*true) - np.mean(pred) * np.mean(true) )
    denom = np.sqrt( (np.mean(true*true) - np.mean(true)*np.mean(true)) * (np.mean(pred*pred) - np.mean(pred)*np.mean(pred)) )
    f = numer / denom
    return f

# 4. Testing
print("Just after train")

# After predicting
y = model.predict(testX_pca)
#y = y_norm * (y_max - y_min) + y_min
print(y.shape)
plt.scatter(testY,y)
plt.xlabel('testY')
plt.ylabel('predY')
plt.plot(np.linspace(0,np.max(testY),20),np.linspace(0,np.max(testY),20),'r')
plt.savefig('test-vs-predict.png')
plt.close()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('diagnose lr')

m = tf.keras.metrics.MeanAbsoluteError()
m.update_state(testY, y)

print(m.result().numpy())
print(fidelity(testY.flatten(), y.flatten()))

np.save("accuracy.npy", m.result().numpy())

model.save("my_model")

loss_array=np.array(history.history['loss'])
vals_array=np.array(history.history['val_loss'])
np.save("loss_array.npy", loss_array)
np.save("vals_array.npy", vals_array)
metrics = evaluate_regression(testY, y)
