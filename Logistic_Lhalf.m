clear;
clc;
%rng(1);
%% Generating simulation data %%
beta=zeros(1,1000);
beta(1)=5;
beta(2)=-5;
beta(3)=5;
beta(4)=-5;
beta(5)=5;
beta(6)=-5;
beta(7)=5;
beta(8)=-5;
beta(9)=5;
beta(10)=-5;

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

train_size=100;
test_size=50;
sample_size=train_size+test_size;

intercept=0.0;
X = normrnd(0, 1, sample_size, size(beta,2)+1);
[n,p]=size(X);

% Setting corrlation %
cor=0;             % correlation %
for i=1:n
    for j=1:p-1
        x(i,j)=X(i,j+1)*sqrt(1-cor)+X(i,1)*sqrt(cor);
    end
end

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

% [Opt,Mse]=CV_Lhalf_logistic(X,Y,Lambda1,beta_path);
% beta=beta_path(:,Opt);
% 
% beta_zero=beta(1);
% beta=beta(2:end);
% l = beta_zero + x_test * beta;
% prob=exp(l)./(1 + exp(l));
% for i=1:test_size
%     if prob(i)>0.5
%         test_y(i)=1;
%     else
%         test_y(i)=0;
%     end
% end
% error=test_y'-y_test;
% count=find(error~=0)
% fail=length(count)
% 
% beta_non_zero=find(beta~=0);
% 
% plot(beta_path','linewidth',1.5)
% ax = axis;
% line([opt opt], [ax(3) ax(4)], 'Color', 'b', 'LineStyle', '-.');
% xlabel('Steps')
% ylabel('Coefficeints')
% 
% figure;
% hold on
% plot(Mse,'linewidth',1.5);
% ax = axis;
% line([opt opt], [ax(3) ax(4)], 'Color', 'b', 'LineStyle', '-.');
% xlabel('Steps')
% ylabel('Misclassification Error')

