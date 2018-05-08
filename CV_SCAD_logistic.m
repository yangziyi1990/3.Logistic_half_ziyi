function [ Opt,Mse ] = CV_SCAD_logistic(X,y,Lambda)

%%%%%%%%%%%%%%     K cross validation    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
k=5;            %%% K-fold%%%
[n,p] = size(X);
valida_n=floor(n/k);
sample_sequence=1:n;

for j=1:length(Lambda)
    lambda=Lambda(j);
    for i=1:k
        if i<=k-1
            validation_seq=sample_sequence(:,(i-1)*valida_n+1:i*valida_n);
        else
            validation_seq=sample_sequence(:,(i-1)*valida_n+1:n);
        end
        train_seq=setdiff(sample_sequence,validation_seq);
        X_train = X(train_seq,:);
        y_train = y(train_seq);
        X_validation= X(validation_seq, :);
        y_validation = y(validation_seq);
        
        col=size(X_train,2);
        row=size(X_train,1);
        beta=zeros(col-1,1);
        temp=sum(y_train)/row;  
        beta_zero=log(temp/(1-temp));   
        beta_int=[beta_zero;beta];
        
        beta=Logistic_SCAD_func(X_train,y_train,beta_int,lambda);
        beta_zero=beta(1);
        beta=beta(2:end);
        X_validation=X_validation(:,2:end);
        test_y = beta_zero + X_validation * beta;
        for m=1:size(y_validation,1)
            if sign(test_y(m))==1
                test_y(m)=1;
            else
                test_y(m)=0;
            end
        end
        error=test_y-y_validation;  
        Mse(i,j)=sum(abs(error));
        test_y=0;
    end
end

[Number,Opt]=min(sum(Mse,1));
Mse=sum(Mse,1);

end


