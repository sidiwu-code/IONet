import numpy as np
from sklearn import gaussian_process as gp
from scipy import interpolate
import os 
data_file = 'data/data025'
print(os.getcwd())
if not os.path.isdir('./'+data_file): os.makedirs('./'+data_file)
length_scalel = 0.25     
length_scaler = 0.25
features = 1000      
sensors_num = 100     


Remark = 'Test'     ### generate training data
#Remark = 'Train'   ### generate test data

if Remark == 'Train':
    
    NNN=10   
    test_save = False 
    train_save = True
    sample_train_num = 10000  

if Remark == 'Test':
    NNN=1
    test_save =  True
    train_save = False 
    sample_train_num = 1000  

############################### mesh ##############################
N =98  
xmin, xmax=0,1
h = (xmax-xmin)/(N+1)
mesh = np.linspace(xmin, xmax, N+2,endpoint=True)  
alpha= 0.5 

def interface_dirich(x):
    return 1.

def interface_numan(x):
    return 0.

def boundary_condition_l():
    return 1

def boundary_condition_r():
    return 0.


def weight_order0(alpha,x_array):
    a=[]
    # Taylor expansion
    a.append([1,1,1])
    a.append(x_array-alpha)
    a.append((x_array-alpha)**2)
    a=np.array(a)
    b=np.array([1,0,0]).reshape(-1,1)
    w=np.dot(np.linalg.inv(a), b)
    return w.reshape(-1,)


def weight_order1(alpha,x_array):
    a=[]
    a.append([1,1,1])
    a.append(x_array-alpha)
    a.append((x_array-alpha)**2)
    a=np.array(a)
    b=np.array([0,1,0]).reshape(-1,1)
    w=np.dot(np.linalg.inv(a), b)
    return w.reshape(-1,)


def fi_and_fiadd1(w01,w02,w11,w12,beta_l,beta_r):
    """
    coef_local: [ui-1,ui,u+1]
    w01,w11: ui-1, ui, ui+1
    w02,w12: ui, ui+1, ui+2
    """
    k1=beta_l
    k2=beta_r
    a=np.array([[w02[0],-w01[2]],[k2*w12[0],-k1*w11[2]]]) 
    a_inv=np.linalg.inv(a)
    b=np.array([[w01[0],w01[1],-w02[1],-w02[2],1,0],[k1*w11[0],k1*w11[1],-k2*w12[1],-k2*w12[2],0,1]]) 
   
    return a_inv@b


def modify_matrix(A,F,mesh,alpha,record,coef,coefx,beta_l,beta_r):
    """
    alpha in i and i+1
    A_local@[ui-1,ui,ui+2,ui+2,[u],[un]].T=[fi,fi+1].T
    """
    if alpha not in mesh:
        
        i=record[0]-1                     
        x_1=mesh[record[0]-1:record[1]+1] # xi-1,xi,xi+1
        x_2=mesh[record[0]:record[1]+2]   # xi, xi+1,xi+2

        w01= weight_order0(alpha,x_1)     # fi+1
        w02= weight_order0(alpha,x_2)     # fi

        w11= weight_order1(alpha,x_1)
        w12= weight_order1(alpha,x_2)

        A_local=fi_and_fiadd1(w01, w02, w11, w12, beta_l, beta_r)


        A[i,i+1] += coef[i+1]                       
        A[i,i-1:i+3]-= A_local[1,:4]*coef[i+1]

        A[i+1,i] += coef[i+2]                        
        A[i+1,i-1:i+3] -= A_local[0,:4]*coef[i+2]
        inter_condi=np.array([interface_dirich(alpha),interface_numan(alpha)]).reshape(-1,1)

        F[i] += (A_local[:,4:]@inter_condi)[1][0]*coef[i+1]
        F[i+1] += (A_local[:,4:]@inter_condi)[0][0]*coef[i+2]

        A[i,i+1] += h/2*coefx[i+1]                   
        A[i,i-1:i+3] -= h/2* A_local[1,:4]*coefx[i+1]

        A[i+1,i] -= h/2*coefx[i+2]
        A[i+1,i-1:i+3] += h/2*A_local[0,:4]*coefx[i+2]  

        F[i] += h/2*(A_local[:,4:]@inter_condi)[1][0]*coefx[i+1]
        F[i+1] -= h/2* (A_local[:,4:]@inter_condi)[0][0]*coefx[i+2]  

        return A,F


def generate_gps():
    x = np.linspace(-0.01, 1.01, num=features)[:, None]
    A = gp.kernels.RBF(length_scale=length_scalel)(x)
    L = np.linalg.cholesky(A + 1e-13 * np.eye(features))
    aa=  L @ np.random.randn(features, sample_train_num)
    gps=(aa-np.min(aa,axis=0,keepdims=True)+1).transpose()

    
    return gps


gps_left = generate_gps()
gps_right = generate_gps()


u_pre_solution = []    
beta_sensor_value = []
betax_sensor_value = []
sensors_location = np.linspace(0, 1, num=sensors_num)

appro_beta_l = []
appro_beta_r = []

xx_l = mesh[::NNN][np.where(mesh[::NNN]<=alpha)[0]]
xx_r = mesh[::NNN][np.where(mesh[::NNN]>alpha)[0]]
beta_train_x = []   
betax_train_x = []  

import time
t0=time.time()
for k in range(sample_train_num):
    alpha_left=interpolate.interp1d(np.linspace(-0.01, 1.01, num=features), gps_left[k], kind='cubic', copy=False, assume_sorted=True) 
    alpha_right=interpolate.interp1d(np.linspace(-0.01, 1.01, num=features), gps_right[k], kind='cubic', copy=False, assume_sorted=True) 
    coef=np.zeros(N+2)
    coefx=np.zeros(N+2)
    index=np.argmin(abs(mesh-alpha))
    tol = 1e-4
    if mesh[index]<=alpha and mesh[index+1]>alpha:
        coef[:index+1]=alpha_left(mesh[:index+1])
        coefx[:index+1] =(alpha_left(mesh[:index+1]+tol)-alpha_left(mesh[:index+1]-tol))/2/tol
        coef[index+1:]=alpha_right(mesh[index+1:])
        coefx[index+1:] =(alpha_right(mesh[index+1:]+tol)-alpha_right(mesh[index+1:]-tol))/2/tol
        record= [index,index+1]
        wl = weight_order0(alpha, mesh[index-1:index+2])
        wr = weight_order0(alpha, mesh[index:index+3])
        beta_inter_l = wl@alpha_left(mesh[index-1:index+2])
        beta_inter_r = wr@alpha_right(mesh[index:index+3])
        
    elif mesh[index]>alpha and mesh[index-1]<=alpha:
        coef[:index]=alpha_left(mesh[:index])
        coefx[:index] =(alpha_left(mesh[:index]+tol)-alpha_left(mesh[:index]-tol))/tol/2
        coef[index:]=alpha_right(mesh[index:])
        coefx[index:] =(alpha_left(mesh[index:]+tol)-alpha_left(mesh[index:]-tol))/2/tol
        record= [index-1,index]
        wl = weight_order0(alpha, mesh[index-2:index+1])
        wr = weight_order0(alpha, mesh[index-1:index+2])
        beta_inter_l = wl@alpha_left(mesh[index-2:index+1])
        beta_inter_r = wr@alpha_right(mesh[index-1:index+2])  
    
    appro_beta_l.append(beta_inter_l)
    appro_beta_r.append(beta_inter_r)
    ###  boundary condition
    u_pre = np.zeros(N+2)
    u_pre[0]= boundary_condition_l()
    u_pre[-1]= boundary_condition_r()

    A1 = np.zeros((N, N))  
    for i in range(N):
        A1[i, i] = -2
        if i < N-1: A1[i, i+1] = 1
        if i > 0:   A1[i, i-1] = 1 
    A1=-np.diag(coef[1:-1])@A1


    A2 = np.zeros((N, N))  
    for i in range(N):
        if i < N-1: A2[i, i+1] = 1
        if i > 0:   A2[i, i-1] = -1 
    A2=-np.diag(coefx[1:-1])@A2*h/2
    A= A1+A2

    F=np.zeros(N).reshape(-1,1)
    index=np.where(mesh[1:-1]<=alpha)[0]
    x_left = mesh[1:-1][index]
    F[index]=(x_left*h**2).reshape(-1,1)*0
    index=np.where(mesh[1:-1]>alpha)[0]
    x_right = mesh[1:-1][index]
    F[index]=(x_right*h**2).reshape(-1,1)*0
    F[0] += u_pre[0]*coef[1]
    F[-1] += u_pre[-1]*coef[-2]

    ## first order
    F[0] -= u_pre[0]*coefx[1]*h/2
    F[-1] += u_pre[-1]*coefx[-2]*h/2

    ### numerical solution
    A_new,F_new=modify_matrix(A,F,mesh,alpha,record,coef,coefx, beta_inter_l, beta_inter_r)  
    u_pre[1:-1]=np.dot(np.linalg.inv(A_new), F_new).reshape(-1,)
    u_pre_solution.append(u_pre[::NNN])


    index = np.where(sensors_location<=alpha)[0]
    beta_left=alpha_left(sensors_location[index])

    index = np.where(sensors_location>alpha)[0]
    beta_right=alpha_right(sensors_location[index])
    beta_sensor_value.append(np.hstack((beta_left,beta_right)))
    
    beta_xl = alpha_left(xx_l)
    betax_xl = (alpha_left(xx_l+tol)-alpha_left(xx_l-tol))/2/tol
    beta_xr = alpha_right(xx_r)
    betax_xr = (alpha_right(xx_r+tol)-alpha_right(xx_r-tol))/2/tol
    
    beta_train_x.append(np.hstack((beta_xl,beta_xr)))
    betax_train_x.append(np.hstack((betax_xl,betax_xr)))

    if k%100==0:
        print(k)


u_pre_solution = np.array(u_pre_solution)      
beta_sensor_value =np.array(beta_sensor_value) 
beta_train_x =np.array(beta_train_x)
betax_train_x = np.array(betax_train_x)

appro_beta_l=np.array(appro_beta_l).reshape(-1,1)
appro_beta_r=np.array(appro_beta_r).reshape(-1,1)



if train_save== True:
    np.savetxt(data_file+'/train_appro_beta_l.txt', appro_beta_l)
    np.savetxt(data_file+'/train_appro_beta_r.txt', appro_beta_r)
    ### sensor
    np.savetxt(data_file+'/train_sensors_location.txt',sensors_location)
    np.savetxt(data_file+'/train_beta_sensor_value.txt',beta_sensor_value.reshape(sample_train_num,-1))
    ### train data points
    np.savetxt(data_file+'/train_mesh.txt',mesh[::NNN])
    np.savetxt(data_file+'/train_u_solution.txt',u_pre_solution.reshape(sample_train_num,-1))
    np.savetxt(data_file+'/train_beta_x.txt',beta_train_x.reshape(sample_train_num,-1))
    np.savetxt(data_file+'/train_betax_x.txt',betax_train_x.reshape(sample_train_num,-1))

if test_save== True:
    ### test data 
    np.savetxt(data_file+'/test_mesh.txt',mesh[::NNN])
    np.savetxt(data_file+'/sensors_location.txt',sensors_location)
    np.savetxt(data_file+'/test_beta_sensor_value.txt', beta_sensor_value.reshape(sample_train_num,-1))
    np.savetxt(data_file+'/test_u_solution.txt',u_pre_solution.reshape(sample_train_num,-1))
