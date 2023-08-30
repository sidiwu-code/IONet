import torch
import torch.nn as nn
import numpy as np
import time 
import os 
from Tools import  get_batch_single
from Net_type import IONet
from Genaratedata_dd_ion import generate_test_data, generate_train_data
import argparse

alpha = np.array(0.5)

def main(args):
    if torch.cuda.is_available() and args.cuda == True:
        device ='cuda'
    else:
        device = 'cpu'
    
    X_test_l, X_test_r,  sensor_num1, sensor_num2= generate_test_data(data_file=args.data_file, alpha=alpha, device=device)
    X_train_l, X_train_r =generate_train_data(data_file=args.data_file, alpha=alpha, device=device, p=args.p)


    model_type = 'IONet'
    width = 100
    model_1 = IONet( m = width, sensor_num1 = sensor_num1,sensor_num2 = sensor_num2, actv = nn.ReLU()).to(device)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model_1.parameters())))

    import torch.optim as optim
    optimizer=optim.Adam(model_1.parameters(),lr=args.lr)
    t0=time.time()
    
    for i in range(args.epoch):
        input_left= get_batch_single(args.batch[0], X_train_l, device=device)
        input_right= get_batch_single(args.batch[1], X_train_r, device=device)
        optimizer.zero_grad()

        loss = nn.MSELoss()(model_1(input_left,label='left'),input_left[-1]) + nn.MSELoss()(model_1(input_right,label='right'),input_right[-1])
        loss.backward(retain_graph=True)
        optimizer.step()

        if i%args.print_epoch==0:
            print('epoch {}: training loss'.format(i), loss.item(),optimizer.param_groups[0]['lr'])
            print('loss: ', loss.item())
            tt = time.time()
            pre_label_l = model_1(X_test_l,label='left')
            pre_label_r = model_1(X_test_r,label='right')
            relative_l2 =((X_test_l[-1]-pre_label_l)**2).sum() +((pre_label_r-X_test_r[-1])**2).sum()
            relative_l2 = torch.sqrt(relative_l2/(((X_test_l[-1])**2).sum()+((X_test_r[-1])**2).sum()))
            
            print('Rela L2 loss is: ', relative_l2.item(),'Test time:',(time.time()-tt)/2)

        if (i+1)%int(args.epoch/100)==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.95


    print('time',time.time()-t0)
    if not os.path.isdir('./'+'model/'): os.makedirs('./'+'model/')
    torch.save(model_1,'model/'+'dd_ion_'+model_type+'_'+str(width)+'_'+str(args.epoch)+'.pkl')



if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_file',type=str, default='data/data025/')
    parser.add_argument('--model_type', type=str, default='Stacked')
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--batch', type=list, default=[1500,1500])
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--epoch', type=int, default=100000)   
    parser.add_argument('--p', type=int, default=10)    # number of train data points of each sample. Note that  ion p/2, deeponet: p
    parser.add_argument('--print_epoch', type=int, default=200)

    args = parser.parse_args()
    main(args)


