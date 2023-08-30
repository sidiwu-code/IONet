import torch
import torch.nn as nn
import numpy as np
import time 
import os 
from Tools import grad, get_batch
from Net_type import DeepONet
import argparse
from Genaratedata_pi_onet import generate_test_data, generate_train_data
import torch.optim as optim

alpha = np.array(0.5)

def main(args):
    if torch.cuda.is_available() and args.cuda == True:
        device ='cuda'
    else:
        device = 'cpu'
            
    width = 140

    ## generate training and test data
    X_test, sensor_num= generate_test_data(data_file=args.data_file, alpha=alpha, device=device)
    train_left,  train_right, train_interface, train_sample_bl, train_sample_br=generate_train_data(data_file=args.data_file, alpha=alpha, device=device, p=args.p)

    model_1 = DeepONet( m = width, sensor_num = sensor_num, actv=nn.Tanh()).to(device)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model_1.parameters())))

    optimizer=optim.Adam(model_1.parameters(),lr=args.lr)
    t0=time.time()
    tol = 1e-5
    for i in range(args.epoch):
        input_left, input_right, input_gamma, input_boundaryl, input_boundaryr= get_batch(args.batch, train_left, train_right, train_interface, train_sample_bl,train_sample_br, device=device)
        optimizer.zero_grad()
        u1= model_1(input_left)
        grad1=grad(u1,input_left[1])
        laplace1=-grad1*input_left[-1] - input_left[-2]*grad(grad1,input_left[1])          
        loss_residual_1 = nn.MSELoss()(laplace1,laplace1*0)

        u2= model_1(input_right)
        grad2=grad(u2,input_right[1])
        laplace2=-grad2*input_right[-1] - input_right[-2]*grad(grad2,input_right[1])    
        loss_residual_2 = nn.MSELoss()(laplace2,laplace2*0)
    
    
        ui_l = model_1(input_gamma,tol=-tol)
        ui_r = model_1(input_gamma,tol= tol)
        loss_interface_d = nn.MSELoss()(ui_l,ui_r-1)

        grad_i1=(model_1(input_gamma)-ui_l)/tol
        grad_i2=(ui_r-model_1(input_gamma))/tol
        loss_interface_n=nn.MSELoss()(grad_i1*input_gamma[-2],grad_i2*input_gamma[-1])

        loss_bd_l = nn.MSELoss()(model_1(input_boundaryl), input_boundaryl[1]*0+1)
        loss_bd_r = nn.MSELoss()(model_1(input_boundaryr), input_boundaryr[1]*0)

        loss = loss_residual_1+loss_residual_2+ (loss_interface_d+loss_interface_n)+ 100*(loss_bd_l+loss_bd_r)
        loss.backward(retain_graph=True)
        optimizer.step()

        if i%args.print_epoch==0:
            print('epoch {}: training loss'.format(i), loss.item())
            print(loss_residual_1.item(),loss_residual_2.item(),loss_interface_d.item(),loss_interface_n.item(),loss_bd_l.item(),loss_bd_r.item())
            pre_label = model_1(X_test)
            relative_l2 =((X_test[-1]-pre_label)**2).sum() 
            relative_l2 = torch.sqrt(relative_l2/((X_test[-1])**2).sum())
            print('Rela L2 loss is: ', relative_l2.item())
            print('\n')

        if (i+1)%int(args.epoch/100)==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.95
      
    print('time',time.time()-t0)
    if not os.path.isdir('./'+'model/'): os.makedirs('./'+'model/')
    torch.save(model_1,'model/'+'pi_onet_'+'_'+str(width)+'_'+str(args.epoch)+'.pkl')



if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_file',type=str, default='data/data025/')
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--batch', type=list, default=[1000,1000,1000])
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--epoch', type=int, default=100000)   
    parser.add_argument('--p', type=int, default=10)    # number of train data points of each sample. Note that  ion p/2, deeponet: p
    parser.add_argument('--print_epoch', type=int, default=500)

    args = parser.parse_args()
    main(args)

