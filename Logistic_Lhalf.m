clear;
clc;
%rng(1);
%% Generating simulation data %%
beta=zeros(1,500);
beta(1)=1;
beta(2)=-1;
beta(3)=1;
beta(4)=-1;
beta(5)=1;
beta(6)=-1;
beta(7)=1;
beta(8)=-1;
beta(9)=1;
beta(10)=-1;

% beta(1)=1.2;
% beta(4)=1.6;
% beta(7)=0.9;
% beta(15)=0.6;
% beta(19)=0.5;
% beta(23)=-1.2;
% beta(26)=1;
% beta(30)=-0.5;
% beta(35)=1.3;
% beta(36)=0.8;

beta_t=beta';
train_size=100;
test_size=20;
sample_size=train_size+test_size;

intercept=0.0;
x = normrnd(0, 1, sample_size, size(beta,2));
[n,p]=size(x);

% Setting corrlation %
% cor=0;             % correlation %
% for i=1:n
%     for j=1:p-1
%         x(i,j)=X(i,j+1)*sqrt(1-cor)+X(i,1)*sqrt(cor);
%     end
% end

%l = intercept + (x * beta' + 0.2 * normrnd(0, 1, n, 1));
l = intercept + x * beta';
prob=exp(l)./(1 + exp(l));

for i=1:sample_size
    if prob(i)>0.5
        y(i)=1;
    else
        y(i)=0;
    end
end
y=y';

x_train=x(1:train_size,:);
x_test=x(train_size+1:sample_size,:);
y_train=y(1:train_size,:);
y_test=y(train_size+1:sample_size,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Logistic + Lhalf %%
col=size(x_train,2);
row=size(x_train,1);
beta=zeros(col,1);

%  calculating the beta_zero  %
temp=sum(y_train)/row;   
beta_zero=log(temp/(1-temp));    

% Inputting X, Y, beta_int and lambda %
beta_int=[beta_zero;beta];
beta_true=[beta_zero;beta_t];
x0=ones(row,1);
X=[x0,x_train];
Y=y_train;

% Step 1: Initialize (u,w,z) %
% u = exp(X * beta_int)./(1 + exp(X * beta_int));
% W = diag(u .* (1 - u));
% z = X * beta_int + inv(W) * (Y - u);
% 
% S = (X' * W * z)/row;
% lambda_max = 0.8; %(4/3 * (max(S)))^(1.5);

lambda_max =norm(X'*Y,'inf'); % according to the https://github.com/yangziyi1990/SparseGDLibrary.git
lambda_min = lambda_max * 0.001;
m=10;
for i=1:m
    Lambda1(i)=lambda_max*(lambda_min/lambda_max)^(i/m);
    lambda=Lambda1(i);
    beta=Logistic_Lhalf_func(X,Y,beta_int,lambda);   
    beta_path(:,i)=beta;
    fprintf('iteration times:%d\n',i);
end

[Opt,Mse]=CV_Lhalf_logistic(X,Y,Lambda1);
beta_opt=beta_path(:,Opt);

beta_zero=beta_opt(1); 
beta=beta_opt(2:end); 
l = beta_zero + x_test * beta;
prob=exp(l)./(1 + exp(l)); 
for i=1:test_size
    if prob(i)>0.5
        test_y(i)=1;
    else
        test_y(i)=0;
    end
end

error=test_y'-y_test;
error_number_testing=length(nonzeros(error))
beta_non_zero=length(nonzeros(beta_opt))

%% Performance
[accurancy,sensitivity,specificity]=performance(y_test,test_y');
fprintf('The accurancy of testing data (lhalf): %f\n' ,accurancy);
fprintf('The sensitivity of testing data (lhalf): %f\n' ,sensitivity);
fprintf('The specificity of testing data (lhalf): %f\n' ,specificity);


%% performance for training data
beta_zero=beta_opt(1); 
beta=beta_opt(2:end); 
l1 = beta_zero + x_train * beta;
prob1=exp(l1)./(1 + exp(l1)); 
for i=1:train_size
    if prob1(i)>0.5
        train_y(i)=1;
    else
        train_y(i)=0;
    end
end
error_train=train_y'-y_train;
error_number_train=length(nonzeros(error_train))

[accurancy_train,sensitivity_train,specificity_train]=performance(y_train,train_y');
fprintf('The accurancy of training data(lhalf): %f\n' ,accurancy_train);
fprintf('The sensitivity of training data (lhalf): %f\n' ,sensitivity_train);
fprintf('The specificity of training data (lhalf): %f\n' ,specificity_train);

%% performance for beta
[accurancy_beta,sensitivity_beta,specificity_beta]=performance_beta(beta_true,beta_opt);
fprintf('The accurancy of beta (lhalf): %f\n' ,accurancy_beta);
fprintf('The sensitivity of beta (lhalf): %f\n' ,sensitivity_beta);
fprintf('The specificity of beta (lhalf): %f\n' ,specificity_beta);
