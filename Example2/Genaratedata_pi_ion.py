import sys
sys.path.append('..')
import numpy as np
from sklearn import gaussian_process as gp
from scipy import interpolate
from Tools import To_tensor

length_scalel = 0.2
length_scaler = 0.1
features = 500
sensors_num = 100
alpha = np.array(0.5)

def generate_test_data(data_file, device):
    
    test_left     = np.loadtxt(data_file+'test_xleft.txt').reshape(-1,1)
    y_left       = np.loadtxt(data_file+'label_left.txt').reshape(-1,1)
    ygrad_left    = np.loadtxt(data_file+'label_grad_left.txt').reshape(-1,1)
    test_right =  np.loadtxt(data_file+'test_xright.txt').reshape(-1,1)
    y_right    = np.loadtxt(data_file+'label_right.txt').reshape(-1,1)
    ygrad_right = np.loadtxt(data_file+'label_grad_right.txt').reshape(-1,1)
    sensor_left = np.loadtxt(data_file+'sensor_left.txt')
    sensor_right = np.loadtxt(data_file+'sensor_right.txt')

    X_test_left = (sensor_left, sensor_right, test_left,  y_left, ygrad_left)
    X_test_right = (sensor_left,sensor_right, test_right, y_right, ygrad_right)

    X_test_left = To_tensor(X_test_left, device)
    X_test_right = To_tensor(X_test_right, device)
    
    return X_test_left, X_test_right


def gaussian_process(sample_num):
    x = np.linspace(0, alpha, num=features)[:, None]
    A_left = gp.kernels.RBF(length_scale=length_scalel)(x)
    L_left = np.linalg.cholesky(A_left + 1e-10 * np.eye(features))
    gps_left=(L_left @ np.random.randn(features, sample_num)).transpose()

    x = np.linspace(alpha,1, num=features)[:, None]
    A_right = gp.kernels.RBF(length_scale=length_scaler)(x)
    L_right = np.linalg.cholesky(A_right+ 1e-10 * np.eye(features))
    gps_right=(L_right @ np.random.randn(features, sample_num)).transpose()

    return  gps_left, gps_right



def generate_train_data(sample_num, device,  p):

    gps1, gps2 = gaussian_process(sample_num)
    
    x_sensor=np.linspace(0, 1, num=sensors_num) 
    xl_sensor = x_sensor[np.where(x_sensor<= alpha)[0]]
    xr_sensor = x_sensor[np.where(x_sensor>alpha)[0]]
    sensor_num2 = len(xr_sensor)
    sensor_num1 = len(xl_sensor)

    interface = np.array([alpha])
    boundary_l = np.array([0.])
    boundary_r = np.array([1.])
    p=int(p/2)
   
    def generate(gp_left,gp_right):  
        x = np.linspace(0, alpha, num=gp_left.shape[-1])[:, None]
        u_left=interpolate.interp1d(x.reshape(-1,), gp_left, kind='cubic', copy=False, assume_sorted=True) 
        x = np.linspace(alpha,1, num=gp_right.shape[-1])[:, None]
        u_right=interpolate.interp1d(x.reshape(-1,), gp_right, kind='cubic', copy=False, assume_sorted=True)
        
        xleft  = np.sort(np.random.rand(p))*alpha 
        xright = np.sort(np.random.rand(p))*(1-alpha)+alpha
        rhsleft=u_left(xleft)
        rhsright=u_right(xright)
        # sensor 
        u_sensors_left = u_left(xl_sensor)
        u_sensors_right = u_right(xr_sensor)            
                    
        sample_left=np.hstack([np.tile(u_sensors_left, (p, 1)), np.tile(u_sensors_right, (p, 1)), xleft[:, None], rhsleft[:,None]])
        sample_right=np.hstack([np.tile(u_sensors_left, (p, 1)), np.tile(u_sensors_right, (p, 1)), xright[:, None], rhsright[:,None]])
        sample_interface=np.hstack([u_sensors_left, u_sensors_right, interface])
        sample_boundaryl = np.hstack([u_sensors_left, u_sensors_right, boundary_l])
        sample_boundaryr = np.hstack([u_sensors_left, u_sensors_right, boundary_r])

        return  sample_left, sample_right, sample_interface, sample_boundaryl, sample_boundaryr
    
    sample_left,sample_right, sample_interface, sample_boundaryl, sample_boundaryr = generate(gps1[0],gps2[0])

    for i in range(1,gps1.shape[0]):
        s1,s2,s3,s4,s5=generate(gps1[i],gps2[i])
        sample_left=np.vstack([sample_left,s1])
        sample_right=np.vstack([sample_right,s2])
        sample_interface = np.vstack([sample_interface,s3])
        sample_boundaryl = np.vstack([sample_boundaryl,s4])
        sample_boundaryr = np.vstack([sample_boundaryr,s5])



    sample_left      =  ( sample_left[...,: sensor_num1],      sample_left[..., sensor_num1: sensor_num1+ sensor_num2],      sample_left[...,-2:-1],sample_left[...,-1:])
    sample_right     =  ( sample_right[...,: sensor_num1],     sample_right[..., sensor_num1: sensor_num1+ sensor_num2],     sample_right[...,-2:-1],sample_right[...,-1:])
    sample_interface =  ( sample_interface[...,: sensor_num1], sample_interface[..., sensor_num1: sensor_num1+ sensor_num2], sample_interface[...,-1:])
    sample_boundaryl =  ( sample_boundaryl[...,: sensor_num1], sample_boundaryl[..., sensor_num1: sensor_num1+ sensor_num2], sample_boundaryl[...,-1:])
    sample_boundaryr =  ( sample_boundaryr[...,: sensor_num1], sample_boundaryr[..., sensor_num1: sensor_num1+ sensor_num2], sample_boundaryr[...,-1:])
    
    sample_left = To_tensor(sample_left, device)
    sample_right = To_tensor(sample_right, device)
    sample_interface = To_tensor(sample_interface, device)
    sample_boundaryl =  To_tensor(sample_boundaryl, device)
    sample_boundaryr = To_tensor(sample_boundaryr, device)

    return sample_left ,sample_right, sample_interface, sample_boundaryl, sample_boundaryr, sensor_num1, sensor_num2
