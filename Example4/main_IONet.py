import numpy as np 
import argparse
import torch
import torch.nn as nn
import time
import itertools
import random
import torch.optim as optim
from Tool import  div,Grad, map_elementwise
from Net_type import IONet
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from GenerateData_IONet import Data
torch.set_default_dtype(torch.float32)


def get_batch(args, out, inner, gamma, input_boundary_up, input_boundary_down, device):
    # batch
    @map_elementwise
    def batch_mask(X, num):
        return np.random.choice(X.size(0), num, replace=False)
    @map_elementwise
    def batch(X, mask):
        return X[mask].requires_grad_(True).float().to(device)

    mask1 = batch_mask(out[-1], args.btrain_out)
    mask2 = batch_mask(inner[-1], args.btrain_inner)
    mask3 = batch_mask(gamma[-1], args.btrain_gamma)
    mask4 = batch_mask(input_boundary_up[-1], args.btrain_boundary)
    mask5 = batch_mask(input_boundary_down[-1], args.btrain_boundary)

    return batch(out, mask1),  batch(inner, mask2), batch(gamma, mask3), batch(input_boundary_up, mask4), batch(input_boundary_down, mask5)


def get_full_batch(args, out, inner, gamma, input_boundary_up, input_boundary_down, device):
    # batch
    @map_elementwise
    def batch(X):
        return X.requires_grad_(True).float().to(device)

    return batch(out),  batch(inner), batch(gamma), batch(input_boundary_up), batch(input_boundary_down)



def batch_type(batch,get_batch=get_batch, get_full_batch=get_full_batch):
    if batch=='full':
        return get_full_batch
    else:
        return get_batch


def main(args):
    if torch.cuda.is_available and args.cuda:
        device='cuda'
        print('cuda is avaliable')
    else:
        device='cpu'

    data = Data(device=device, gap=args.gap)
    # train data
    TX_train_up, TX_train_down, TX_train_interface, TX_train_b_up, TX_train_b_down = data.Generate_train_data()
    # test data
    x_up, label_up, x_down, label_down = data.Generate_test_data()
    sensor_num_up = data.sensors_nup
    sensor_num_down = data.sensors_ndown

    ### select batch of training data 
    batch_function = batch_type(args.batch)
    input_up, input_down, input_interface, input_boundary_up, input_boundary_down = batch_function(args, TX_train_up, TX_train_down, TX_train_interface, TX_train_b_up, TX_train_b_down,device)
    
    model=IONet(sensor_up=sensor_num_up, sensor_down=sensor_num_down, m=args.unit).to(device)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

    optimizer=optim.Adam(itertools.chain(model.parameters()),lr=args.lr)
    t0=time.time()


    if not os.path.isdir('./outputs/'+args.filename+'/model'): os.makedirs('./outputs/'+args.filename+'/model')

    for epoch in range(args.nepochs):
        optimizer.zero_grad()
        U1=model(input_up,label='up')
        grads_in=Grad(U1,input_up[2])
        div_in = -div(grads_in,input_up[2])
        loss_up=torch.mean((div_in)**2)  
        
        U1_b=model(input_interface,label='up')
        U2_b_in=model(input_interface,label='down')          
        loss_gammad=torch.mean((U1_b-U2_b_in)**2)  

        dU1_N=torch.autograd.grad(U1_b,input_interface[2], grad_outputs=torch.ones_like(U1_b).to(device), create_graph=True)[0]      
        U1_N=dU1_N[:,1].view(-1,1)       
        dU2_N=torch.autograd.grad(U2_b_in,input_interface[2], grad_outputs=torch.ones_like(U1_b).to(device), create_graph=True)[0]
        U2_N=dU2_N[:,1].view(-1,1)*2      
        loss_gamman=torch.mean((U1_N-U2_N)**2)   

        U2 = model(input_down,label='down') 
        grads_out=Grad(U2,input_down[2])
        div_out = -div(grads_out,input_down[2])*2        
        loss_down=torch.mean((div_out)**2)
        
        loss_boundary_up=torch.mean((model(input_boundary_up,label='up')-input_boundary_up[3])**2)
        loss_boundary_down=torch.mean((model(input_boundary_down,label='down')-input_boundary_down[3])**2)

        loss = loss_up+ loss_down + loss_gammad +loss_gamman + 100*(loss_boundary_up + loss_boundary_down)
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # resample
        input_up, input_down, input_interface, input_boundary_up, input_boundary_down = batch_function(args, TX_train_up, TX_train_down, TX_train_interface, TX_train_b_up, TX_train_b_down,device)
        if (epoch+1)%args.print_num==0:
 
            with torch.no_grad():

                Mse_train=(loss_up+loss_down + loss_gammad +loss_gamman + loss_boundary_up + loss_boundary_down).item()      
                print('Epoch,  Training MSE          : ',epoch+1,Mse_train)
                print('        loss up/down          : ',loss_up.item(), loss_down.item())
                print('        loss interd/intern    : ',loss_gammad.item(), loss_gamman.item())
                print('        loss boundup/bounddown: ',loss_boundary_up.item(), loss_boundary_down.item())
                
                test_label = model(x_up,label='up').view(100,-1)
                label_up = label_up.view(100,-1)
                L2_up_loss = torch.sqrt(((test_label-label_up)**2).sum(dim=-1)/((label_up)**2).sum(dim=-1))
       
                test_label = model(x_down,label='down').view(100,-1)
                label_down = label_down.view(100,-1)
                L2_down_loss = torch.sqrt(((test_label-label_down)**2).sum(dim=-1)/((label_down)**2).sum(dim=-1))
           
                
                print('Test Losses')
                print('        Relative L2 up(mean/std)   :',torch.mean(L2_up_loss).item(), torch.std(L2_up_loss).item())  
                print('        Relative L2 down(mean/std) :',torch.mean(L2_down_loss).item(), torch.std(L2_down_loss).item())                                                                                 
                print('*****************************************************')
        
                if epoch>args.nepochs*0.95:             
                    torch.save(model, 'outputs/'+args.filename+'/model/gap_{}_model.pkl'.format(args.gap))
                   
        if  (epoch+1)%int(args.nepochs/100)==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.95
            
    print('totle use time:',time.time()-t0)
        
if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser() 
    parser.add_argument('--filename',type=str, default='result')
    parser.add_argument('--unit', type=int, default=120)
    parser.add_argument('--gap', type=int, default=8) 

    parser.add_argument('--batch',type=str, default='batch') 
    parser.add_argument('--btrain_inner', type=int, default=2000)
    parser.add_argument('--btrain_out', type=int, default=2000)  
    parser.add_argument('--btrain_gamma', type=int, default=2000)
    parser.add_argument('--btrain_boundary', type=int, default=2000) 

    parser.add_argument('--print_num', type=int, default=500)
    parser.add_argument('--nepochs', type=int, default=100000)   
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--save', type=str, default=False)

    args = parser.parse_args()
    main(args)

        
            
