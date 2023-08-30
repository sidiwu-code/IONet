%The number of training samples
N = 500;

%Number of grid points on [0,1]^2 
s = 257; 
[X,Y] = meshgrid(0:(1/(s-1)):1);

%Parameters of covariance C = tau^(alpha/2)*(-Laplacian + tau^2 I)^(-alpha)
%Laplacian has zero Neumann boundry
alpha = 4; 
tau = 10;

train_data=zeros(N,4*s-4); 
for i = 1: N
    % up£ºy>0.5
    norm_a = GRF(alpha, tau, s);
    % down
    tmp = GRF(alpha, tau, s);
    norm_a(1:ceil((s+1)/2),:)=tmp(1:ceil((s+1)/2),:);
    
    tmp = norm_a(:,1)';   
    tmp = cat(2,tmp,norm_a(s,2:s-1)); 
    tmp = cat(2,tmp,norm_a(s:-1:1,s)'); 
    tmp = cat(2,tmp,norm_a(1,s-1:-1:2)); 
    
    train_data(i,:) = tmp;
end
csvwrite('./train_data/train_data_samples.csv',train_data) % dim is (N, 4*s-4)

x_points = X(:,1)';   
x_points = cat(2,x_points,X(s,2:s-1)); 
x_points = cat(2,x_points,X(s:-1:1,s)'); 
x_points = cat(2,x_points,X(1,s-1:-1:2)); 

y_points = Y(:,1)';   
y_points = cat(2,y_points,Y(s,2:s-1)); 
y_points = cat(2,y_points,Y(s:-1:1,s)'); 
y_points = cat(2,y_points,Y(1,s-1:-1:2)); 

xy_datas  =cat(1,x_points,y_points)';
csvwrite('./sample_boundary_points.csv',xy_datas) % dim is (4*s-4,2)