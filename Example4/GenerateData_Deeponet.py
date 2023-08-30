import torch
import numpy as np
torch.set_default_dtype(torch.float32)

class Data(object):
    def __init__(self, device,gap):
        self.device = device

        self.sample_bounary_points = torch.tensor(np.loadtxt(open('GRF-generate/sample_boundary_points.csv','rb'),delimiter=',')).float().to(device)  # num*dim
        self.train_domain_num = 500              
        self.train_interface_num = 500           
        self.train_boundary_num = 500          
        self.box = [0,1,0,1]        
        self.gap = gap                         

    def Generate_test_data(self):
        test_data = self.__Generate_test_data()  
        print('The number of testing data:')
        print('     X_test_up/down       : {}/{}'.format(test_data[0].shape[0],test_data[0].shape[0]))
        print('------------------------------------------------------------------------------')

        return test_data
    

    def Generate_train_data(self):
        X_train = self.__Generate_train_data()

        print('The number of training data:')
        print('     X_train   shape      : {}/{}'.format(X_train[0].shape[0], X_train[1].shape[0]))
        print('------------------------------------------------------------------------------')

        return X_train

    def __Generate_train_data(self): 
        sample_boundary_features = torch.tensor(np.loadtxt(open('GRF-generate/train_data/train_data_samples.csv','rb'),delimiter=',')).float().to(self.device)
        sample_boundary_features =sample_boundary_features[:,0:-1:self.gap]
        self.sensor_num = sample_boundary_features.shape[1]


        train_points = torch.tensor(np.loadtxt('GRF-generate/train_data/train_points.txt', dtype=np.float32)).to(self.device)
        train_labels =  torch.tensor(np.loadtxt('GRF-generate/train_data/train_Nlabels.txt', dtype=np.float32)).float().to(self.device) # (10, 289)
        
        for i in range(sample_boundary_features.shape[0]):
            if i==0:
                X = torch.hstack((torch.tile(sample_boundary_features[i], (train_points.shape[0], 1)),  train_points, train_labels[i].view(-1,1)))
            else:
                tmpup = torch.hstack((torch.tile(sample_boundary_features[i], (train_points.shape[0], 1)), train_points,train_labels[i].view(-1,1)))
                X = torch.cat((X,tmpup),dim=0)
 
        X = (X[..., :self.sensor_num], X[..., -3:-1], X[..., -1:])

        return X


    def __Generate_test_data(self): 
        sample_boundary_features = torch.tensor(np.loadtxt(open('GRF-generate/test_data/test_data_samples.csv','rb'),delimiter=',')).float().to(self.device)
        sample_boundary_features =sample_boundary_features[:,0:-1:self.gap]
        self.sensor_num = sample_boundary_features.shape[1]
       
        test_points = torch.tensor(np.loadtxt('GRF-generate/test_data/test_points.txt', dtype=np.float32)).to(self.device)
        test_labels =  torch.tensor(np.loadtxt('GRF-generate/test_data/test_Nlabels.txt', dtype=np.float32)).float().to(self.device)  #FEM
        

        for i in range(sample_boundary_features.shape[0]):
            if i==0:
                up = torch.hstack((torch.tile(sample_boundary_features[i], (test_points.shape[0], 1)), test_points, test_labels[i].view(-1,1)))
            else:
                tmpup = torch.hstack((torch.tile(sample_boundary_features[i], (test_points.shape[0], 1)), test_points,test_labels[i].view(-1,1)))
                up = torch.cat((up,tmpup),dim=0)

       
        x_test = (up[..., :self.sensor_num], up[..., -3:-1],up[...,-1:])


        return x_test
