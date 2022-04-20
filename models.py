from math import nan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 4
ORDER = 3
AR_ORDER = 1

def make_vector(dataframe):
    numpy_array = dataframe.to_numpy()
    vector_data = []

    for i in range(len(dataframe.values)):
        for column in range(len(dataframe.columns)):
            print(numpy_array[i][column])
            vector_data.append(numpy_array[i][column])
    
    return np.array(vector_data, dtype='float')

def arx(y, u, data, word_count, y_index):
    form = []
    # form.append([y])
    for j in range(AR_ORDER):
        form.append([y[y_index - j - 1]])

    for i in range(ORDER * word_count):
        form.append([data[u - i - word_count]])
    
    return np.array(form, dtype='float')

def ewls(data, t, word_count, y_data):
    j = 0
    Y = [[]]
    R = 0
    p = 0
    exp_lambda = 0.9
    Y_array = []

    for i in range(ORDER - 1, t):
        w = pow(exp_lambda, i)
        R += w * arx(y_data, i, data, word_count, j) @ arx(y_data, i, data, word_count, j).T
        p += w * y_data[j] * arx(y_data, i, data, word_count, j)

        if np.linalg.det(R) != 0:
            ewls_estimator = np.linalg.inv(R) @ p
            Y = arx(y_data, i, data, word_count, j).T @ ewls_estimator
            Y_array.append(Y[0][0])
        else:
            Y_array.append(np.mean(Y_array))
        
        if j < len(y_data) - 1:
            j += 1

    Y_array = np.nan_to_num(Y_array, nan=0.0)
    return Y_array

def progressive_arx(y, u, data, word_count, y_index, prog_order):
    form = []
    
    if prog_order <= AR_ORDER:
        for j in range(AR_ORDER):
            form.append([y[y_index - j - 1 - prog_order]])
            print("Prog order for AR", prog_order)

    if prog_order > AR_ORDER:
        for i in range(ORDER * word_count):
            form.append([data[u - i - word_count - prog_order]])
            print("U prog", prog_order)
    
    return np.array(form, dtype='float')

def stationary_ls(data, t, word_count, y_data):
    j = 0
    Y = [[]]
    Y_array = []
    
    for prog_order in range(ORDER * word_count + AR_ORDER):
        epsilon = 0
        R = 0
        p = 0
        for i in range(ORDER - 1, t):
            R += progressive_arx(y_data, i, data, word_count, j, prog_order) @ progressive_arx(y_data, i, data, word_count, j, prog_order).T
            p += y_data[j] * progressive_arx(y_data, i, data, word_count, j, prog_order)

            if np.linalg.det(R) != 0:
                ls_estimator = np.linalg.inv(R) @ p
                Y = progressive_arx(y_data, i, data, word_count, j, prog_order).T * ls_estimator
                Y_array.append(Y[0][0])
            else:
                Y_array.append(np.mean(Y_array))

            epsilon += (y_data[j - 1] - Y_array[j - 1]) ** 2

        if j < len(y_data) - 1:
            j += 1
    
    Y_array = np.nan_to_num(Y_array, nan=0.0)
    return Y_array

def akaike_fpe():
    pass
