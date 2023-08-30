import numpy as np
from Genaratedata_dd_onet import generate_test_data, generate_train_data
import torch
import torch.nn as nn
import time 
import os
import argparse
from Tools import grad,  get_batch_single
from Net_type import DeepONet
import torch.optim as optim


alpha = np.array([0.5])

def main(args):
    if torch.cuda.is_available() and args.cuda == True:
        device ='cuda:0'
    else:
        device = 'cpu'

    width = 140
    sensor_num =100
    
    ## generate training and test data
    X_test = generate_test_data(data_file=args.data_file,  device=device)
    X_train = generate_train_data(sample_test_num=args.sample_num, device=device, p=args.p)
    model_1 = DeepONet(m=width, sensor_num = sensor_num, actv=nn.ReLU()).to(device)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model_1.parameters())))


    optimizer=optim.Adam(model_1.parameters(),lr=args.lr)
    t0=time.time()
    for i in range(args.epoch):
        input= get_batch_single(args.batch, X_train, device=device)
        optimizer.zero_grad()
        loss = nn.MSELoss()( model_1(input),input[-2])
        loss.backward(retain_graph=True)
        optimizer.step()

        if i%args.print_epoch==0:
            print('epoch {}: training loss'.format(i), loss.item(),optimizer.param_groups[0]['lr'])
            ### 
            pre_label = model_1(X_test).view(args.sample_num,-1)
            true_label = X_test[-2].view(args.sample_num,-1)
            relative_l2 =((true_label-pre_label)**2).sum(dim=-1) 
            relative_l2 = torch.sqrt(relative_l2/((true_label)**2).sum(dim=-1))

            ## grad
            grad_label=X_test[-1].view(args.sample_num,-1)
            grad_pre = grad(model_1(X_test),X_test[1])
            loss_grad = ((grad_label-grad_pre)**2).sum(dim=-1)
            loss_grad = torch.sqrt(loss_grad/((grad_pre)**2).sum(dim=-1))

            print('Rela L2 loss/ grad error: ', relative_l2.item(), loss_grad.item())
            print('\n')

        if (i+1)%int(args.epoch/100)==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.95

    print('time',time.time()-t0)
    if not os.path.isdir('./'+'model/'): os.makedirs('./'+'model/')
    torch.save(model_1,'model/'+'pi_ion_'+'_'+str(width)+'_'+str(args.epoch)+'.pkl')


if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_file',type=str, default='data/data_01_02/')
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--batch', type=list, default=30)
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--epoch', type=int, default=400)  
    parser.add_argument('--sample_num', type=int, default=100)   
    parser.add_argument('--p', type=int, default=10)  
    parser.add_argument('--print_epoch', type=int, default=200)


    args = parser.parse_args()
    main(args)


