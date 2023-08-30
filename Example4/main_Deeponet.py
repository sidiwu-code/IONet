import numpy as np 
import argparse
import torch
import torch.nn as nn
import time
import itertools
import random
import torch.optim as optim
from Tool import  map_elementwise
from Net_type import DeepONet
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from GenerateData_Deeponet import Data
torch.set_default_dtype(torch.float32)



def get_batch(args, x,  device):
    @map_elementwise
    def batch_mask(X, num):
        return np.random.choice(X.size(0), num, replace=False)
    @map_elementwise
    def batch(X, mask):
        return X[mask].requires_grad_(True).float().to(device)

    mask1 = batch_mask(x[-1], args.btrain_train)

    return batch(x, mask1)


def get_full_batch(args,  x, device):
    @map_elementwise
    def batch(X):
        return X.requires_grad_(True).float().to(device)

    return batch(x)



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
    TX_train = data.Generate_train_data()
    # test data
    TX_test = data.Generate_test_data()
    # number of sensors
    sensor_num = data.sensor_num

   
    batch_function = batch_type(args.btrain_train)
    
    model=DeepONet(sensor_num=sensor_num,  m=args.unit).to(device)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    optimizer=optim.Adam(itertools.chain(model.parameters()),lr=args.lr)
    t0=time.time()
    if not os.path.isdir('./outputs/'+args.filename+'/model'): os.makedirs('./outputs/'+args.filename+'/model')

    for epoch in range(args.nepochs):
        input_x  = batch_function(args, TX_train, device)
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(input_x),input_x[-1].view(-1,1))
        loss.backward(retain_graph=True)
        optimizer.step()

        if (epoch+1)%args.print_num==0:

            with torch.no_grad():
                print('Epoch,  Training MSE          : ',epoch+1,loss.item())
                # rela L_2
                test_label = model(TX_test).view(100,-1)
                label_up = TX_test[-1].view(100,-1)
                L2_up_loss = torch.sqrt(((test_label-label_up)**2).sum(dim=-1)/((label_up)**2).sum(dim=-1))
                print(torch.mean(L2_up_loss).item(), torch.std(L2_up_loss).item())
                
                print('Test Losses')
                print('        Relative L2 up(mean/std    :',torch.mean(L2_up_loss).item(), torch.std(L2_up_loss).item())                                                                             
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
    parser.add_argument('--unit', type=int, default=170)
    parser.add_argument('--gap', type=int, default=8)  
    parser.add_argument('--batch',type=str, default='batch')
    parser.add_argument('--btrain_train', type=int, default=10000)

    parser.add_argument('--print_num', type=int, default=500)
    parser.add_argument('--nepochs', type=int, default=100000)   
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--save', type=str, default=False)

    args = parser.parse_args()
    main(args)

        
            
