import torch
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


def generate_train_data(data_file, alpha, device , p):
    boundary_l = np.array(0.).reshape(-1,1)
    boundary_r = np.array(1.).reshape(-1,1)
    
    sensor_location = np.loadtxt(data_file+'sensors_location.txt')
    sensor_left_index = np.where(sensor_location<=alpha)[0]
    sensor_right_index = np.where(sensor_location>alpha)[0]
    sensor_num1 = len(sensor_left_index)                
    sensor_num2 = len(sensor_right_index)                
   
    train_beta_sensor_value=np.loadtxt(data_file+'train_beta_sensor_value.txt')
    train_beta_sensor_l = train_beta_sensor_value[:,sensor_left_index]
    train_beta_sensor_r = train_beta_sensor_value[:,sensor_right_index]

    train_mesh_location = np.loadtxt(data_file+'train_mesh.txt')
    x_index_l = np.where(train_mesh_location<=alpha)[0]
    x_index_r = np.where(train_mesh_location>alpha)[0]
    x_left = train_mesh_location[x_index_l]
    x_right = train_mesh_location[x_index_r]

    xx_beta = np.loadtxt(data_file+'train_beta_x.txt')
    xx_beta_l = xx_beta[:,x_index_l]
    xx_beta_r = xx_beta[:,x_index_r]
    xx_betax = np.loadtxt(data_file+'train_betax_x.txt')#
    xx_betax_l = xx_betax[:,x_index_l]
    xx_betax_r = xx_betax[:,x_index_r]        

    appro_beta_l = np.loadtxt(data_file+'train_appro_beta_l.txt').reshape(-1,1)
    appro_beta_r = np.loadtxt(data_file+'train_appro_beta_r.txt').reshape(-1,1)

    pp=int(p/2)

    def generate(s_l, s_r, b_x_l, b_x_r, bx_x_l, bx_x_r, betal, betar):  
        x_index_left = random.sample(range(len(x_left)), pp)
        sample_left=np.hstack([np.tile(s_l, ( pp, 1)), 
                                np.tile(s_r, ( pp, 1)),
                                x_left[x_index_left].reshape(-1,1),
                                b_x_l[x_index_left].reshape(-1,1),
                                bx_x_l[x_index_left].reshape(-1,1)])

        x_index_right = random.sample(range(len(x_right)), pp)
        sample_right=np.hstack([np.tile(s_l, ( pp, 1)), 
                                np.tile(s_r, ( pp, 1)),
                                x_right[x_index_right].reshape(-1,1),
                                b_x_r[x_index_right].reshape(-1,1),
                                bx_x_r[x_index_right].reshape(-1,1)])
        
        sample_interface=np.hstack([np.tile(s_l, (1, 1)), 
                                    np.tile(s_r, (1, 1)),
                                    alpha.reshape(-1,1),
                                    betal.reshape(-1,1),
                                    betar.reshape(-1,1)])

        sample_boundaryl = np.hstack([np.tile(s_l, (1, 1)), 
                                    np.tile(s_r, (1, 1)),
                                    boundary_l.reshape(-1,1)])
        
        sample_boundaryr = np.hstack([np.tile(s_l, (1, 1)), 
                                    np.tile(s_r, (1, 1)),
                                    boundary_r.reshape(-1,1)])

        return  sample_left, sample_right, sample_interface, sample_boundaryl, sample_boundaryr


    for i in range(0,train_beta_sensor_value.shape[0]):
        if i%200==0:
            print(i)
        if i==0:
            sample_left, sample_right, sample_interface, sample_boundaryl, sample_boundaryr = generate(train_beta_sensor_l[i],train_beta_sensor_r[i],xx_beta_l[i], xx_beta_r[i], xx_betax_l[i], xx_betax_r[i],appro_beta_l[i],appro_beta_r[i])
        else:
            s1,s2,s3,s4,s5 = generate(train_beta_sensor_l[i],train_beta_sensor_r[i],xx_beta_l[i], xx_beta_r[i], xx_betax_l[i], xx_betax_r[i],appro_beta_l[i],appro_beta_r[i])
            sample_left=np.vstack([sample_left,s1])
            sample_right=np.vstack([sample_right,s2])
            sample_interface = np.vstack([sample_interface,s3])
            sample_boundaryl = np.vstack([sample_boundaryl,s4])
            sample_boundaryr = np.vstack([sample_boundaryr,s5])

    sample_left      =  ( sample_left[...,:sensor_num1],      sample_left[...,sensor_num1:sensor_num1+sensor_num2],      sample_left[...,-3:-2],sample_left[...,-2:-1],sample_left[...,-1:])   ## 最后三列有
    sample_right     =  ( sample_right[...,:sensor_num1],     sample_right[...,sensor_num1:sensor_num1+sensor_num2],     sample_right[...,-3:-2],sample_right[...,-2:-1],sample_right[...,-1:])
    sample_interface =  ( sample_interface[...,:sensor_num1], sample_interface[...,sensor_num1:sensor_num1+sensor_num2], sample_interface[...,-3:-2], sample_interface[...,-2:-1], sample_interface[...,-1:])
    sample_boundaryl =  ( sample_boundaryl[...,:sensor_num1], sample_boundaryl[...,sensor_num1:sensor_num1+sensor_num2], sample_boundaryl[...,-1:])
    sample_boundaryr =  ( sample_boundaryr[...,:sensor_num1], sample_boundaryr[...,sensor_num1:sensor_num1+sensor_num2], sample_boundaryr[...,-1:])

    train_left = To_tensor(sample_left, device)
    train_right = To_tensor(sample_right, device)
    train_interface = To_tensor(sample_interface, device)
    train_sample_bl =  To_tensor(sample_boundaryl, device)
    train_sample_br = To_tensor(sample_boundaryr, device)
    
    return train_left,  train_right, train_interface, train_sample_bl, train_sample_br
