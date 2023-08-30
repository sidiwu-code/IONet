import sys
sys.path.append('..')
import numpy as np
from sklearn import gaussian_process as gp
from scipy import interpolate
import random
import os
from Tools import To_tensor

length_scalel = 0.2
length_scaler = 0.1
features = 500
sensors_num = 100

k1=1
k2=2

N = 998                
xmin,xmax= 0,1        
h = (xmax-xmin)/(N+1)
mesh = np.linspace(xmin, xmax, N+2,endpoint=True)  
alpha= 0.5           

def f_dirich(x):
    return 0.

def f_numan(x):
    return 0.


def weight_order0(alpha,x_array):
    """
    zero order
    """
    a=[]
    a.append([1,1,1])
    a.append(x_array-alpha)
    a.append((x_array-alpha)**2)
    a=np.array(a)
    b=np.array([1,0,0]).reshape(-1,1)
    w=np.dot(np.linalg.inv(a), b)

    return w.reshape(-1,)


def weight_order1(alpha,x_array):
    """
    first order
    """
    a=[]
    a.append([1,1,1])
    a.append(x_array-alpha)
    a.append((x_array-alpha)**2)
    a=np.array(a)
    b=np.array([0,1,0]).reshape(-1,1)
    w=np.dot(np.linalg.inv(a), b)

    return w.reshape(-1,)


def fi_and_fiadd1(w01,w02,w11,w12):
    """
    return: A_local
            A_local@[ui-1,ui,ui+2,ui+2,[u],[un]] = [f_{i},f_{i+1}]
    """
    a=np.array([[w02[0],-w01[2]],[k2*w12[0],-k1*w11[2]]]) 
    a_inv=np.linalg.inv(a)
    b=np.array([[w01[0],w01[1],-w02[1],-w02[2],1,0],[k1*w11[0],k1*w11[1],-k2*w12[1],-k2*w12[2],0,1]]) #ui-1,ui,ui+2,ui+2,[u],[un]
    return a_inv@b


def modify_matrix(A,F,mesh,alpha,record):
    """
    alpha in i and i+1 
    A_local@[ui-1,ui,ui+2,ui+2,[u],[un]].T=[fi,fi+1].T  
    """
    if alpha not in mesh:
        i=record[0]-1                     
        x_1=mesh[record[0]-1:record[1]+1] 
        x_2=mesh[record[0]:record[1]+2]  
        w01= weight_order0(alpha,x_1)
        w02= weight_order0(alpha,x_2)

        w11= weight_order1(alpha,x_1)
        w12= weight_order1(alpha,x_2)

        A_local=fi_and_fiadd1(w01,w02,w11,w12)

        A[i,i+1]=0                        
        A[i,i-1:i+3]-= A_local[1,:4]*k1  

        A[i+1,i]=0                       
        A[i+1,i-1:i+3]-= A_local[0,:4]*k2
        inter_condi= np.array([f_dirich(alpha),f_numan(alpha)]).reshape(-1,1)

        F[i]+=(A_local[:,4:]@inter_condi)[1][0]*k1
        F[i+1]+=(A_local[:,4:]@inter_condi)[0][0]*k2

    else: 

        index=np.argmin(abs(mesh[1:-1]-alpha))     
        F[index+1]-=k2*f_dirich(alpha)      

        x_1=mesh[1:-1][index-1:index+2]  # ui-1,ui,fi+1
        x_2=mesh[1:-1][index:index+3]    # fi,xi+1,xi+2

        w11= weight_order1(alpha,x_1)*k1
        w12= weight_order1(alpha,x_2)*k2        

        # [ui-1,ui,ui+1,ui+2]
        cc=np.array([-w11[0],-w11[1]+w12[0],w12[1],w12[2]])/w11[2]*k1
        A[index,index+1]=0
        A[index,index-1:index+3]+=cc
        F[index] -= f_numan(alpha)/w11[2]*k1
        F[index] -= w12[0]*f_dirich(alpha)/w11[2]*k1

    return A,F

def deal_with_mesh(mesh=mesh, alpha=alpha, k1=k1,k2=k2):
    """
    input: mesh, alpha, k1, k2
    output:record and coef 
    """   
    coef=np.zeros(N+2)
    index=np.argmin(abs(mesh-alpha))
    if mesh[index]<=alpha and mesh[index+1]>alpha:
        coef[:index+1]=k1
        coef[index+1:]=k2
        record= [index,index+1]
        
    elif mesh[index]>alpha and mesh[index-1]<=alpha:
        coef[:index]=k1
        coef[index:]=k2
        record= [index-1,index]

    return np.array(record), np.array(coef) 

record, coef = deal_with_mesh()


def generate_test_data(data_file, device):
    
    test_left     = np.loadtxt(data_file+'test_xleft.txt').reshape(-1,1)
    y_left       = np.loadtxt(data_file+'label_left.txt').reshape(-1,1)
    ygrad_left    = np.loadtxt(data_file+'label_grad_left.txt').reshape(-1,1)
    test_right =  np.loadtxt(data_file+'test_xright.txt').reshape(-1,1)
    y_right    = np.loadtxt(data_file+'label_right.txt').reshape(-1,1)
    ygrad_right = np.loadtxt(data_file+'label_grad_right.txt').reshape(-1,1)
    sensor_left = np.loadtxt(data_file+'sensor_left.txt')
    sensor_right = np.loadtxt(data_file+'sensor_right.txt')

    x_test = np.vstack((test_left,test_right))
    sensors = np.hstack((sensor_left,sensor_right))
    sensors = np.vstack((sensors,sensors))
    u_label = np.vstack((y_left,y_right))
    u_grad_label = np.vstack((ygrad_left,ygrad_right))

    X_test =(sensors, x_test, u_label, u_grad_label)
    X_test = To_tensor(X_test, device)
    
    return X_test


def solve(gp_left, gp_right, xl_sensor, xr_sensor, p, alpha=alpha,features=features,mesh=mesh):
    x = np.linspace(0, alpha, num=features)[:, None]
    u_left=interpolate.interp1d(x.reshape(-1,), gp_left, kind='cubic', copy=False, assume_sorted=True) 
    x = np.linspace(alpha,1, num=features)[:, None]
    u_right=interpolate.interp1d(x.reshape(-1,), gp_right, kind='cubic', copy=False, assume_sorted=True)

    # boundary condition
    u_pre = np.zeros(N+2)
    u_pre[0]= 0.
    u_pre[-1]= 0.


    # matrix
    A = np.zeros((N, N))  
    for i in range(N):
        A[i, i] = 2
        if i < N-1: A[i, i+1] = -1
        if i > 0:   A[i, i-1] = -1 

    A=np.diag(coef[1:-1])@A
    
    #### rhs
    F=np.zeros(N).reshape(-1,1)
    index=np.where(mesh[1:-1]<=alpha)[0]
    F[index]=(u_left(mesh[1:-1][index])*h**2).reshape(-1,1)
    index=np.where(mesh[1:-1]>alpha)[0]
    F[index]=(u_right(mesh[1:-1][index])*h**2).reshape(-1,1)
    F[0] += u_pre[0]*k1
    F[-1] += u_pre[-1]*k2


    # mib
    A_new,F_new=modify_matrix(A,F,mesh,alpha,record)
    u_pre[1:-1]=np.dot(np.linalg.inv(A_new), F_new).reshape(-1,)

    # grad
    grad_u_pre=[]
    g1=(u_pre[1]*(-5/2)+u_pre[2]*4+u_pre[3]*(-3/2))/h
    grad_u_pre.append(g1)
    for i in range(1,N+1):              
        g1=(u_pre[i+1]-u_pre[i-1])/(2*h) # O(h^2)
        grad_u_pre.append(g1)
    g1=(u_pre[-2]*(5/2)+u_pre[-3]*(-4)+u_pre[-4]*(3/2))/h
    grad_u_pre.append(g1)
    grad_u_pre=np.array(grad_u_pre)


    # test
    index = random.sample(range(1000), p)
    x_train=mesh[index]
    label = u_pre[index]
    grad_label = grad_u_pre[index]
    
    # sensor
    sensorl=np.tile(u_left(xl_sensor), (x_train.shape[0], 1))
    sensorr=np.tile(u_right(xr_sensor), (x_train.shape[0], 1))
    sensor_values=np.hstack((sensorl,sensorr))

    return sensor_values, x_train.reshape(-1,1), label.reshape(-1,1), grad_label.reshape(-1,1)


def generate_train_data( sample_test_num, device, p):
    N = 998
    xmin,xmax= 0,1
    h = (xmax-xmin)/(N+1)
    mesh = np.linspace(xmin, xmax, N+2,endpoint=True)  
    ### define the interface

    alpha= 0.5

    length_scalel = 0.2
    length_scaler = 0.1
    features = 500
    ## sensor number 
    sensors_num = 100 

    x = np.linspace(0, alpha, num=features)[:, None]
    A_left = gp.kernels.RBF(length_scale=length_scalel)(x)
    L_left = np.linalg.cholesky(A_left + 1e-10 * np.eye(features))
    gps_left=(L_left @ np.random.randn(features, sample_test_num)).transpose()

    x = np.linspace(alpha,1, num=features)[:, None]
    A_right = gp.kernels.RBF(length_scale=length_scaler)(x)
    L_right = np.linalg.cholesky(A_right+ 1e-10 * np.eye(features))
    gps_right=(L_right @ np.random.randn(features, sample_test_num)).transpose()

    x_sensor=np.linspace(0, 1, num=sensors_num) 
    xl_sensor = x_sensor[np.where(x_sensor<=alpha)[0]]
    xr_sensor = x_sensor[np.where(x_sensor>alpha)[0]]

    sensor_values, x_train, label,grad_label = solve(gps_left[0],gps_right[0], xl_sensor=xl_sensor, xr_sensor=xr_sensor,p=p)
  
    for i in range(1, sample_test_num):
        aa,bb,cc ,dd= solve(gps_left[i],gps_right[i], xl_sensor=xl_sensor, xr_sensor=xr_sensor,p=p)
        sensor_values=np.vstack((sensor_values, aa))
        x_train=np.vstack((x_train,bb))
        label=np.vstack((label,cc))
        grad_label =np.vstack((grad_label,dd))

    X_train = (sensor_values, x_train, label,grad_label)
    X_train = To_tensor(X_train,device =device)

    
    return X_train

    
