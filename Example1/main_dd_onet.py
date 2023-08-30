import torch
import torch.nn as nn
import numpy as np
import time 
import random
import os 
import argparse
from Tools import  get_batch_single
from Net_type import DeepONet
from Genaratedata_dd_onet import generate_test_data, generate_train_data
import torch.optim as optim

alpha = np.array(0.5)

def main(args):
    
    if torch.cuda.is_available() and args.cuda == True:
        device ='cuda'
    else:
        device = 'cpu'

    width = 140

    ## generate training and test data
    X_test,  sensor_num = generate_test_data(data_file=args.data_file, alpha=alpha, device=device)
    X_train, sensor_num = generate_train_data(data_file=args.data_file, alpha=alpha, device=device, p=args.p)
    #sensor_num = 100

    ## choose models
    model_1 = DeepONet( m = width, sensor_num = sensor_num).to(device)
    model_type = 'onet'
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model_1.parameters())))

    optimizer=optim.Adam(model_1.parameters(),lr=args.lr)

    t0=time.time()
    for i in range(args.epoch):
        batch_X_train= get_batch_single(args.batch, X_train, device=device)
        
        optimizer.zero_grad()
        loss=nn.MSELoss()(model_1(batch_X_train),batch_X_train[-1])
        
        loss.backward(retain_graph=True)
        optimizer.step()  
        

        if i%args.print_epoch==0:
            print('epoch {}: training loss'.format(i), loss.item())

            relative_l2_error = torch.sqrt(((model_1(X_test)-X_test[-1])**2).sum()/((X_test[-1])**2).sum())
            print('Rela L2 loss is: ', relative_l2_error.item())
            print('\n')

        if (i+1)%int(args.epoch/100)==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.95

    print('time',time.time()-t0)
    if not os.path.isdir('./'+'model/'): os.makedirs('./'+'model/')
    torch.save(model_1,'model/'+'dd_ont_'+model_type+'_'+str(width)+'_'+str(args.epoch)+'.pkl')



if __name__ == '__main__':
    torch.cuda.set_device(1)
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_file',type=str, default='data/data025/')
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--batch', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--epoch', type=int, default=100000)   
    parser.add_argument('--p', type=int, default=10)    # number of train data points of each sample. Note that  ion p/2, deeponet: p
    parser.add_argument('--print_epoch', type=int, default=500)
    
    args = parser.parse_args()
    main(args)
