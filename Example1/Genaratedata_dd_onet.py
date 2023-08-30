import torch
import numpy as np
import random
from Tools import To_tensor

def generate_test_data(data_file, alpha, device):
    test_beta_sensor_value = np.loadtxt(data_file+'/test_beta_sensor_value.txt')
    sensor_num = test_beta_sensor_value.shape[1] 
    test_mesh_location = np.loadtxt(data_file+'/test_mesh.txt')
    test_u_pre_solution = np.loadtxt(data_file+'/test_u_solution.txt')

    for i in range(test_beta_sensor_value.shape[0]):
        if i==0:
            x_test = np.hstack([np.tile(test_beta_sensor_value[i], (test_mesh_location.shape[0], 1)), 
                                test_mesh_location[:,None],
                                test_u_pre_solution[i][:,None]])
        else:
            ll = np.hstack([np.tile(test_beta_sensor_value[i], (test_mesh_location.shape[0], 1)), 
                                test_mesh_location[:,None],
                                test_u_pre_solution[i][:,None]])

            x_test  = np.vstack([x_test, ll])


    X_test_data = (x_test[:,:  sensor_num],  x_test[:,-2][:,None], x_test[:,-1][:,None])
    X_test_data = To_tensor(X_test_data, device)


    return X_test_data, sensor_num


def generate_train_data(data_file, alpha, device , p):
  
    sensor_location = np.loadtxt(data_file+'/sensors_location.txt')
    sensor_num = len(sensor_location)                

    train_beta_sensor_value=np.loadtxt(data_file+'/train_beta_sensor_value.txt')

    train_mesh_location = np.loadtxt(data_file+'/train_mesh.txt')
    train_mesh_label = np.loadtxt(data_file+'/train_u_solution.txt')
 
    pp = p

    for i in range(train_mesh_label.shape[0]):
        index = random.sample(range(len(train_mesh_location)), pp)
        if i==0:
            x_train = np.hstack([np.tile(train_beta_sensor_value[i], (pp, 1)), 
                                train_mesh_location[index][:,None],train_mesh_label[i][index][:,None]])
        else:
            ll = np.hstack([np.tile(train_beta_sensor_value[i], (pp, 1)), 
                                train_mesh_location[index][:,None],train_mesh_label[i][index][:,None]])

            x_train  = np.vstack([x_train, ll])


    X_train_data = (x_train[:,:  sensor_num], x_train[:,-2][:,None], x_train[:,-1][:,None])


    X_train_data = To_tensor(X_train_data, device)

    
    return X_train_data,  sensor_num
