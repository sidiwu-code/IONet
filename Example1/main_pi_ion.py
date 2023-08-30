import torch
import torch.nn as nn
import numpy as np
import time 
import os
import argparse
from Tools import grad, get_batch
from Net_type import IONet
from Genaratedata_pi_ion import generate_test_data, generate_train_data
import torch.optim as optim

alpha = np.array([0.5])

def main(args):
    if torch.cuda.is_available() and args.cuda == True:
        device ='cuda'
    else:
        device = 'cpu'
    
    ## generate training and test data
    X_test_l, X_test_r,  sensor_num1, sensor_num2= generate_test_data(data_file=args.data_file, alpha=alpha, device=device)
    train_left,  train_right, train_interface, train_sample_bl, train_sample_br=generate_train_data(data_file=args.data_file, alpha=alpha, device=device, p=args.p)

    model_type = 'IONet'
    width = 100
    model_1 = IONet( m = width, sensor_num1 = sensor_num1,sensor_num2 = sensor_num2).to(device)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model_1.parameters())))

    optimizer=optim.Adam(model_1.parameters(),lr=args.lr)
    t0=time.time()
    
    for i in range(args.epoch):
        input_left, input_right, input_gamma, input_boundaryl, input_boundaryr= get_batch(args.batch, train_left, train_right, train_interface, train_sample_bl,train_sample_br, device=device)
        optimizer.zero_grad()
        u1= model_1(input_left,label='left')
        grad1=grad(u1,input_left[2])
        laplace1=-grad1*input_left[-1] - input_left[-2]*grad(grad1,input_left[2])         
        loss_residual_1 = nn.MSELoss()(laplace1,laplace1*0)

        u2= model_1(input_right,label='right')
        grad2=grad(u2,input_right[2])
        laplace2=-grad2*input_right[-1] - input_right[-2]*grad(grad2,input_right[2])      
        loss_residual_2 = nn.MSELoss()(laplace2,laplace2*0)
    
        ui_l = model_1(input_gamma,label='left')
        ui_r = model_1(input_gamma,label='right')
        loss_interface_d = nn.MSELoss()(ui_l,ui_r-1)

        z=torch.ones(input_gamma[0].size()[0]).view(-1,1).to(input_gamma[0].device)
        grad_i1=torch.autograd.grad(ui_l,input_gamma[2], grad_outputs=z, create_graph=True)[0]
        grad_i2=torch.autograd.grad(ui_r,input_gamma[2], grad_outputs=z, create_graph=True)[0] 
        loss_interface_n=nn.MSELoss()(grad_i1*input_gamma[-2],grad_i2*input_gamma[-1])

        loss_bd_l = nn.MSELoss()(model_1(input_boundaryl,label='left'), input_boundaryl[2]*0+1)
        loss_bd_r = nn.MSELoss()(model_1(input_boundaryr,label='right'), input_boundaryr[2]*0)

        loss = loss_residual_1+loss_residual_2+ loss_interface_d+loss_interface_n + 100*(loss_bd_l+loss_bd_r)
        loss.backward(retain_graph=True)
        optimizer.step()

        if i%args.print_epoch==0:
            print('epoch {}: training loss'.format(i), loss.item(),optimizer.param_groups[0]['lr'])
            print(loss_residual_1.item(),loss_residual_2.item(),loss_interface_d.item(),loss_interface_n.item(),loss_bd_l.item(),loss_bd_r.item())
            pre_label_l = model_1(X_test_l,label='left')
            pre_label_r = model_1(X_test_r,label='right')
            relative_l2 =((X_test_l[-1]-pre_label_l)**2).sum() +((pre_label_r-X_test_r[-1])**2).sum()
            relative_l2 = torch.sqrt(relative_l2/(((X_test_l[-1])**2).sum()+((X_test_r[-1])**2).sum()))
            print('Rela L2 loss is: ', relative_l2.item())
            print('\n')

        if (i+1)%int(args.epoch/100)==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.95

    print('time',time.time()-t0)
    if not os.path.isdir('./'+'model/'): os.makedirs('./'+'model/')
    torch.save(model_1,'model/'+'pi_ion_'+model_type+'_'+str(width)+'_'+str(args.epoch)+'.pkl')


if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_file',type=str, default='data/data025/')
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--batch', type=list, default=[1000,1000,1000])
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--epoch', type=int, default=100000)   
    parser.add_argument('--p', type=int, default=10)    # number of train data points of each sample. Note that  ion p/2, deeponet: p
    parser.add_argument('--print_epoch', type=int, default=200)
    args = parser.parse_args()
    main(args)

