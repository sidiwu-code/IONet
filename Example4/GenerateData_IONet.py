import torch
import random
import numpy as np
torch.set_default_dtype(torch.float32)

class Data(object):
    def __init__(self, device, gap):
        self.device = device

        self.sample_bounary_points = torch.tensor(np.loadtxt(open('GRF-generate/sample_boundary_points.csv','rb'),delimiter=',')).float().to(device)  
        self.train_domain_num = 500              
        self.train_interface_num = 500           
        self.train_boundary_num = 500           
        self.box = [0,1,0,1]        
        self.gap = gap                            

  
    def Generate_test_data(self):
        test_up, test_label_up, test_down,test_label_down = self.__Generate_test_data()  
        print('The number of testing data:')
        print('     X_test_up/down       : {}/{}'.format(test_label_up.shape[0],test_label_down.shape[0]))
        print('     Sensor up/down       : {}/{}'.format(test_up[0].shape[1], test_up[1].shape[1]))
        print('------------------------------------------------------------------------------')

        return test_up, test_label_up, test_down,test_label_down

    def Generate_train_data(self):
        train_u, train_d = self.__gaussian_process_train()  
        X_train_up, X_train_down, X_train_interface, X_train_b_up, X_train_b_down = self.__Generate_train_data(train_u,train_d) 
        
        print('The number of training data:')
        print('     X_train_up/down      : {}/{}'.format(X_train_up[0].shape[0], X_train_down[0].shape[0]))
        print('     X_train_interface    : {}'.format(X_train_interface[0].shape[0]))
        print('     X_boundary_up/down   : {}/{}'.format(X_train_b_up[0].shape[0], X_train_b_down[0].shape[0]))    
        print('------------------------------------------------------------------------------')

        return X_train_up, X_train_down, X_train_interface, X_train_b_up, X_train_b_down 


#######################################################################################################################################
    def __Generate_train_data(self, gps_u, gps_d): 
       
        def generate(gp_u,gp_d):
            xb_up, labelb_up, xb_down, labelb_down = self.SampleFromBoundary(gp_u,gp_d)
            x_i=self.SampleFrominterface()
            x_up,x_down = self.SampleFromDomain()
            u_sensors_up = gp_u[0:-1: self.gap]             
            u_sensors_down = gp_d[0:-1: self.gap]
            self.sensors_nup=u_sensors_up.size()[0]
            self.sensors_ndown=u_sensors_down.size()[0]

            sample1 = torch.hstack((torch.tile(u_sensors_up, (self.train_domain_num, 1)), torch.tile(u_sensors_down, (self.train_domain_num, 1)), x_up))
            sample2 = torch.hstack((torch.tile(u_sensors_up, (self.train_domain_num, 1)), torch.tile(u_sensors_down, (self.train_domain_num, 1)), x_down))
            sample3 = torch.hstack((torch.tile(u_sensors_up, (self.train_interface_num, 1)), torch.tile(u_sensors_down, (self.train_interface_num, 1)), x_i))
            sample4 = torch.hstack((torch.tile(u_sensors_up, (xb_up.shape[0], 1)), torch.tile(u_sensors_down, (xb_up.shape[0], 1)), xb_up, labelb_up))
            sample5 = torch.hstack((torch.tile(u_sensors_up, (xb_down.shape[0], 1)), torch.tile(u_sensors_down, (xb_down.shape[0], 1)), xb_down, labelb_down))
            
            return sample1.to(self.device),sample2.to(self.device),sample3.to(self.device),sample4.to(self.device),sample5.to(self.device)
           

        for i in range(gps_u.shape[0]): 
            if i==0:
                sample1,sample2,sample3,sample4,sample5=generate(gps_u[0],gps_d[0])
            else:
                s1,s2,s3,s4,s5=generate(gps_u[i],gps_d[i])
                sample1= torch.vstack((sample1,s1))
                sample2= torch.vstack((sample2,s2))
                sample3= torch.vstack((sample3,s3))
                sample4= torch.vstack((sample4,s4))
                sample5= torch.vstack((sample5,s5))    

 
        sample_domain1= (sample1[..., :self.sensors_nup], sample1[..., self.sensors_nup:self.sensors_nup+self.sensors_ndown], sample1[..., -2:]) 
        sample_domain2= (sample2[..., :self.sensors_nup], sample2[..., self.sensors_nup:self.sensors_nup+self.sensors_ndown], sample2[..., -2:]) 
        sample_domain3= (sample3[..., :self.sensors_nup], sample3[..., self.sensors_nup:self.sensors_nup+self.sensors_ndown], sample3[..., -2:])
        sample_domain4= (sample4[..., :self.sensors_nup], sample4[..., self.sensors_nup:self.sensors_nup+self.sensors_ndown], sample4[..., -3:-1],sample4[..., -1].view(-1,1))
        sample_domain5= (sample5[..., :self.sensors_nup], sample5[..., self.sensors_nup:self.sensors_nup+self.sensors_ndown], sample5[..., -3:-1],sample5[..., -1].view(-1,1))

        return sample_domain1,sample_domain2,sample_domain3,sample_domain4,sample_domain5

    def __gaussian_process_train(self): 
        """
        sample from gps for training process
        """
    
        sample_data = torch.tensor(np.loadtxt(open('GRF-generate/train_data/train_data_samples.csv','rb'),delimiter=',')).to(self.device) 
        index_up = torch.where(self.sample_bounary_points[:,1]>0.5)[0]
        index_down = torch.where(self.sample_bounary_points[:,1]<=0.5)[0]
        sample_up = sample_data[:,index_up]        
        sample_down =sample_data[:,index_down]    

        return sample_up.to(self.device), sample_down.to(self.device)


    def __Generate_test_data(self): 
        """
        sample from gps for test process
        """ 
        sample_boundary_features = torch.tensor(np.loadtxt(open('GRF-generate/test_data/test_data_samples.csv','rb'),delimiter=',')).float().to(self.device)
        sample_bounary_points = torch.tensor(np.loadtxt(open('GRF-generate/sample_boundary_points.csv','rb'),delimiter=',')).float()
        ind_up = torch.where(sample_bounary_points[:,1]>0.5)[0]
        ind_down = torch.where(sample_bounary_points[:,1]<=0.5)[0]
        u_sensors_up=(sample_boundary_features[:,ind_up])[:,0:-1:self.gap]   
        u_sensors_down=(sample_boundary_features[:,ind_down])[:,0:-1:self.gap]  
       
 
        test_points = torch.tensor(np.loadtxt('GRF-generate/test_data/test_points.txt', dtype=np.float32)).to(self.device)
        test_labels =  torch.tensor(np.loadtxt('GRF-generate/test_data/test_Nlabels.txt', dtype=np.float32)).float().to(self.device) # (10, 289)
        index_up =  torch.where(test_points[:,1]>0.5)[0]
        index_down = torch.where(test_points[:,1]<=0.5)[0]
        test_points_up = test_points[index_up,:]
        test_points_down = test_points[index_down,:]

        for i in range(sample_boundary_features.shape[0]):
            if i==0:
                up = torch.hstack((torch.tile(u_sensors_up[i], (test_points_up.shape[0], 1)), torch.tile(u_sensors_down[i], (test_points_up.shape[0], 1)), test_points_up))
                down = torch.hstack((torch.tile(u_sensors_up[i], (test_points_down.shape[0], 1)), torch.tile(u_sensors_down[i], (test_points_down.shape[0], 1)), test_points_down))
                label_up = test_labels[i][index_up].view(-1,1)
                label_down = test_labels[i][index_down].view(-1,1)
            else:
                tmpup = torch.hstack((torch.tile(u_sensors_up[i], (test_points_up.shape[0], 1)), torch.tile(u_sensors_down[i], (test_points_up.shape[0], 1)), test_points_up))
                tmpdown = torch.hstack((torch.tile(u_sensors_up[i], (test_points_down.shape[0], 1)), torch.tile(u_sensors_down[i], (test_points_down.shape[0], 1)), test_points_down))
                tmplabel_up = test_labels[i][index_up].view(-1,1)
                tmplabel_down = test_labels[i][index_down].view(-1,1)
                up = torch.cat((up,tmpup),dim=0)
                down = torch.cat((down,tmpdown),dim=0)
                label_up = torch.cat((label_up,tmplabel_up),dim=0)
                label_down = torch.cat((label_down,tmplabel_down),dim=0)
       
        x_up = (up[..., :self.sensors_nup], up[..., self.sensors_nup:self.sensors_nup+self.sensors_ndown], up[..., -2:])
        x_down = (down[..., :self.sensors_nup], down[..., self.sensors_nup:self.sensors_nup+self.sensors_ndown], down[..., -2:])
        return x_up, label_up, x_down, label_down

    def SampleFrominterface(self):
  
        x = torch.rand(self.train_interface_num).view(-1,1)
        y = torch.ones_like(x)*0.5
        X= torch.cat((x,y),dim=1)  

        return X.to(self.device) 

    def __sampledomain(self):
        xmin,xmax,ymin,ymax=self.box
        x = torch.rand(3*self.train_domain_num).view(-1,1) * (xmax - xmin) + xmin
        y = torch.rand(3*self.train_domain_num).view(-1,1) * (ymax - ymin) + ymin
        X =torch.cat((x,y),dim=1)
        
        return X.to(self.device)


    def SampleFromDomain(self):
        """
        training: up and down
        """
        X=self.__sampledomain()      
        location=torch.where(X[:,1]>0.5)[0]
        X_up=X[location,:]
        X_up=X_up[:self.train_domain_num,:]
  
        location=torch.where(X[:,1]<=0.5)[0]
        X_down=X[location,:]      
        X_down = X_down[:self.train_domain_num,:]

        return X_up.to(self.device),  X_down.to(self.device)



    def SampleFromBoundary(self,gp_u,gp_d):
         
        index_up = torch.where(self.sample_bounary_points[:,1]>0.5)[0]
        index_down = torch.where(self.sample_bounary_points[:,1]<=0.5)[0]

        index = random.sample(range(0,len(index_up)), self.train_boundary_num)
        x_up = (self.sample_bounary_points[index_up,:])[index,:]
        label_up = gp_u[index]

        index = random.sample(range(0,len(index_down)), self.train_boundary_num)
        x_down = (self.sample_bounary_points[index_down,:])[index,:]
        label_down = gp_d[index]

        return x_up, label_up.view(-1,1), x_down, label_down.view(-1,1)
