import numpy as np 
import argparse
import torch
import time,os
import itertools
import torch.optim as optim
from Tool import  Grad, div, map_elementwise
from Net_type import  IONet
from GenerateData import Data


def u(x,label):
    """
    exact solution
    """
    x=x.t()  
    if label=='inner':
        u=(1/(1+10*(x[0]**2+x[1]**2))).view(-1,1)
    elif label=='out':
        u=(2/(1+10*(x[0]**2+x[1]**2))).view(-1,1)
    else:
        raise ValueError("invalid label for u(x)")
   
    return u


def inter_dirich(x):
    x=x.t()
    return (1/(1+10*(x[0]**2+x[1]**2))).view(-1,1)


def test_data_net(args,device):  
    
    step=0.02
    x = np.arange(-1, 1+step, step)
    y = np.arange(-1, 1+step, step)
    xx,yy=np.meshgrid(x,y)
    input_x=torch.tensor(xx).view(-1,1).to(device)
    input_y=torch.tensor(yy).view(-1,1).to(device)
    input=(torch.cat((input_x,input_y),1)).float()
    x=input[:,0]
    y=input[:,1]

    rr=np.cbrt(x.cpu().numpy())**2+np.cbrt(y.cpu().numpy())**2
    rr=torch.tensor(rr) 
    r=0.65**(2/3)
    location=torch.where(rr<r)[0]
    test_inner=(input[location,:])
    location=torch.where(rr>r)[0]
    test_out=(input[location,:])
    label_out=u(test_out,label='out')
    label_inner=u(test_inner,label='inner')

    return test_out.to(device),label_out.to(device),test_inner.to(device),label_inner.to(device)


def get_batch(args, out, inner, gamma, input_boundary, device):
    # batch
    @map_elementwise
    def batch_mask(X, num):
        return np.random.choice(X.size(0), num, replace=False)
    @map_elementwise
    def batch(X, mask):
        return X[mask].requires_grad_(True).float().to(device)

    mask1 = batch_mask(out[-2], args.btrain_out)
    mask2 = batch_mask(inner[-2], args.btrain_inner)
    mask3 = batch_mask(gamma[-2], args.btrain_gamma)
    mask4 = batch_mask(input_boundary[-2], args.btrain_boundary)

    return batch(out, mask1),  batch(inner, mask2), batch(gamma, mask3), batch(input_boundary, mask4), 


def get_full_batch(args, out, inner, gamma, input_boundary_in, device):
    # batch
  
    @map_elementwise
    def batch(X):
        return X.requires_grad_(True).float().to(device)

    return batch(out),  batch(inner), batch(gamma), batch(input_boundary_in)


def batch_type(batch, get_batch=get_batch, get_full_batch=get_full_batch):
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
        
    ### train data
    data=Data(device=device)

    # test data
    test_inner, label_inner, test_out, label_out = data.Generate_test_data()

    # train data 
    TX_train_in, TX_train_out, TX_train_interface, TX_train_b = data.Generate_train_data()
    batch_function = batch_type(args.batch)
    input_in, input_out, input_gamma, input_boundary = batch_function(args, TX_train_in, TX_train_out, TX_train_interface, TX_train_b, device)

    model=IONet(sensor_in=data.num_sensor_in,  sensor_out=data.num_sensor_out,  m=args.unit).to(device)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

    optimizer=optim.Adam(itertools.chain(model.parameters()),lr=args.lr)
    t0=time.time()
    if not os.path.isdir('./outputs/'+args.filename+'/model'): os.makedirs('./outputs/'+args.filename+'/model')


    loss_history =[]
    for epoch in range(args.nepochs):      
        optimizer.zero_grad()    
        U1=model(input_in,label='inner')
        grads_in= Grad(U1,input_in[2])
        div_in = -div(grads_in,input_in[2])
        loss_in=torch.mean((2*div_in-input_in[-1])**2)

        U1_b = model(input_gamma , label='inner')
        U2_b = model(input_gamma , label='out')        
        loss_gammad=torch.mean((U2_b-U1_b-inter_dirich(input_gamma[2]))**2)

        dU1_N=torch.autograd.grad(U1_b,input_gamma[2], grad_outputs=torch.ones_like(U1_b).to(device), create_graph=True)[0]     
        dU2_N=torch.autograd.grad(U2_b,input_gamma[2], grad_outputs=torch.ones_like(U1_b).to(device), create_graph=True)[0]
        G_NN=((dU2_N-2*dU1_N)*input_gamma[-1]).sum(dim=1).view(-1,1)   
        loss_gamman=torch.mean((G_NN)**2)   
        
        U2 = model(input_out,  label='out')
        grads_out = Grad(U2,input_out[2])
        div_out = -div(grads_out,input_out[2])      
        loss_out=torch.mean((div_out-input_out[-1])**2)
        loss_boundary=torch.mean((model(input_boundary,  label='out')-input_boundary[-1])**2)

        loss = loss_in + loss_out + loss_gamman + loss_gammad + 100*loss_boundary
        loss.backward(retain_graph=True)
        optimizer.step()

        input_in, input_out, input_gamma, input_boundary = batch_function(args, TX_train_in, TX_train_out, TX_train_interface, TX_train_b, device)

                
        if (epoch+1)%args.print_num==0:
            with torch.no_grad():    
                Mse_train=(loss_in+loss_out+loss_boundary+loss_gammad+loss_gamman).item()      
                print('Epoch,  Training MSE: ',epoch+1,Mse_train,optimizer.param_groups[0]['lr'])
                # rela L_2
                l2_error = ((model(test_inner,label='inner')-label_inner)**2).sum()+((model(test_out,label='out')-label_out)**2).sum()
                l2_error =  torch.sqrt(l2_error/(((label_inner)**2).sum()+((label_out)**2).sum()))

                # L_infty
                L_inf_inner_loss=torch.max(torch.abs(model(test_inner,label='inner')-label_inner))
                L_inf_out_loss=torch.max(torch.abs(model(test_out,label='out')-label_out))
                print('L_infty:',max(L_inf_inner_loss.item(),L_inf_out_loss.item()))
                print('Rel. L_2:',l2_error.item())
                print('*****************************************************')  

                if epoch>args.nepochs*0.98:             
                    torch.save(model, 'outputs/'+args.filename+'/model/model.pkl')

                if args.save:
                    loss_history.append([epoch,loss_in.item(),loss_gammad.item(),loss_out.item(),loss_boundary.item(),loss_gamman.item()])
                    loss_record = np.array(loss_history)
                    np.savetxt('outputs/'+args.filename+'/loss_record.txt', loss_record)           

        if  (epoch+1)%int(args.nepochs/100)==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.95



    print('Totle training time:',time.time()-t0)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--filename',type=str, default='results')
    parser.add_argument('--unit', type=int, default=50)

    # batch of training data
    parser.add_argument('--batch',type=str, default='batch')
    parser.add_argument('--btrain_inner', type=int, default=2000)
    parser.add_argument('--btrain_out', type=int, default=2000)  
    parser.add_argument('--btrain_gamma', type=int, default=2000)
    parser.add_argument('--btrain_boundary', type=int, default=2000) 

    parser.add_argument('--print_num', type=int, default=400)
    parser.add_argument('--nepochs', type=int, default=40000)
    parser.add_argument('--lr', type=float, default=0.001)       
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--box', type=list, default=[-1,1,-1,1])
    parser.add_argument('--save', type=str, default=False)
    args = parser.parse_args()
    main(args)

           
