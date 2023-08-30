import numpy as np
import random
from Tools import To_tensor

def generate_test_data(data_file, alpha, device):
    sensor_location = np.loadtxt(data_file+'sensors_location.txt')
    test_beta_sensor_value = np.loadtxt(data_file+'test_beta_sensor_value.txt')


    sensor_left_index = np.where(sensor_location <= alpha)[0]
    sensor_right_index = np.where(sensor_location > alpha)[0]   
    sensor_num1 = len(sensor_left_index)
    sensor_num2 = len(sensor_right_index)    

    sensor_left = test_beta_sensor_value[:,sensor_left_index]
    sensor_right = test_beta_sensor_value[:,sensor_right_index]

    test_mesh_location = np.loadtxt(data_file+'test_mesh.txt')
    test_u_pre_solution = np.loadtxt(data_file+'test_u_solution.txt')
    x_index_l = np.where(test_mesh_location<= alpha)[0]
    x_index_r = np.where(test_mesh_location> alpha)[0]

    x_l = test_mesh_location[x_index_l]
    x_r = test_mesh_location[x_index_r]
    label_l = test_u_pre_solution[:,x_index_l]
    label_r = test_u_pre_solution[:,x_index_r]

    for i in range(test_beta_sensor_value.shape[0]):
        if i==0:
            x_left = np.hstack([np.tile(sensor_left[i], (x_l.shape[0], 1)), 
                                np.tile(sensor_right[i], (x_l.shape[0], 1)),
                                x_l[:,None],label_l[i][:,None]])
            
            x_right = np.hstack([np.tile(sensor_left[i], (x_r.shape[0], 1)), 
                                np.tile(sensor_right[i], (x_r.shape[0], 1)),
                                x_r[:,None],label_r[i][:,None]])
        else:
            ll = np.hstack([np.tile(sensor_left[i], (x_l.shape[0], 1)), 
                                np.tile(sensor_right[i], (x_l.shape[0], 1)),
                                x_l[:,None],label_l[i][:,None]])
            
            rr = np.hstack([np.tile(sensor_left[i], (x_r.shape[0], 1)), 
                                np.tile(sensor_right[i], (x_r.shape[0], 1)),
                                x_r[:,None],label_r[i][:,None]])

            x_left  = np.vstack([x_left, ll])
            x_right = np.vstack([x_right, rr])

    X_test_left = (x_left[:,:  sensor_num1], x_left[:,  sensor_num1: sensor_num1+ sensor_num2], x_left[:,-2][:,None], x_left[:,-1][:,None])
    X_test_right = (x_right[:,:  sensor_num1], x_right[:,  sensor_num1: sensor_num1+ sensor_num2], x_right[:,-2][:,None], x_right[:,-1][:,None])


    X_test_l = To_tensor(X_test_left, device)
    X_test_r = To_tensor(X_test_right, device)

    return X_test_l, X_test_r, sensor_num1, sensor_num2


def generate_train_data(data_file, alpha, device, p):
    sensor_location = np.loadtxt(data_file+'sensors_location.txt')
    train_beta_sensor_value = np.loadtxt(data_file+'train_beta_sensor_value.txt')


    sensor_left_index = np.where(sensor_location <= alpha)[0]
    sensor_right_index = np.where(sensor_location > alpha)[0]   
    sensor_num1 = len(sensor_left_index)
    sensor_num2 = len(sensor_right_index)    

    sensor_left = train_beta_sensor_value[:,sensor_left_index]
    sensor_right = train_beta_sensor_value[:,sensor_right_index]

    train_mesh_location = np.loadtxt(data_file+'train_mesh.txt')
    train_u_pre_solution = np.loadtxt(data_file+'train_u_solution.txt')
    x_index_l = np.where(train_mesh_location<= alpha)[0]
    x_index_r = np.where(train_mesh_location> alpha)[0]

    x_l = train_mesh_location[x_index_l]
    x_r = train_mesh_location[x_index_r]
    label_l = train_u_pre_solution[:,x_index_l]
    label_r = train_u_pre_solution[:,x_index_r]

    p=int(p/2)
    for i in range(train_beta_sensor_value.shape[0]):
        
        if i==0:
            index = random.sample(range(len(x_l)), p)
            x_left = np.hstack([np.tile(sensor_left[i], (p, 1)), 
                                np.tile(sensor_right[i], (p, 1)),
                                x_l[index][:,None],label_l[i][index][:,None]])
            index = random.sample(range(len(x_r)), p)
            x_right = np.hstack([np.tile(sensor_left[i], (p, 1)), 
                                np.tile(sensor_right[i], (p, 1)),
                                x_r[index][:,None],label_r[i][index][:,None]])
        else:
            index = random.sample(range(len(x_l)), p)
            ll = np.hstack([np.tile(sensor_left[i], (p, 1)), 
                                np.tile(sensor_right[i], (p, 1)),
                                x_l[index][:,None],label_l[i][index][:,None]])
            index = random.sample(range(len(x_r)), p)
            rr = np.hstack([np.tile(sensor_left[i], (p, 1)), 
                                np.tile(sensor_right[i], (p, 1)),
                                x_r[index][:,None],label_r[i][index][:,None]])

            x_left  = np.vstack([x_left, ll])
            x_right = np.vstack([x_right, rr])

    X_train_left = (x_left[:,:  sensor_num1], x_left[:,  sensor_num1: sensor_num1+ sensor_num2], x_left[:,-2][:,None], x_left[:,-1][:,None])
    X_train_right = (x_right[:,:  sensor_num1], x_right[:,  sensor_num1: sensor_num1+ sensor_num2], x_right[:,-2][:,None], x_right[:,-1][:,None])


    X_train_l = To_tensor(X_train_left, device)
    X_train_r = To_tensor(X_train_right, device)

    return X_train_l, X_train_r

