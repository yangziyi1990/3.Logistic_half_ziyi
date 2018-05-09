function beta = Logistic_Lhalf_func(X,Y,beta_int,lambda)

    col=size(X,2);
    row=size(X,1);
% Step 1: Initialize (u,w,z) %
    u = exp(X * beta_int)./(1 + exp(X * beta_int));
    W = diag(u .* (1 - u));
    z = X * beta_int + pinv(W) * (Y - u);

% Step 2: The coordinate descent algorithm for sparse logistic with the L1/2 penalty %
    iter=1;
    maxiter=100;
    beta=beta_int;
    beta_old=ones(col,1);
          
while iter<=maxiter && norm(beta_old - beta) > (1E-8)
    t_start = tic;
    
    beta_old=beta;
    beta_zero=sum(W *( z - X(:,2:end) * beta(2:end)))/sum(sum(W)); 
    for k=2:col
        C(k)=(X(:,k)' * W * (z - beta_zero - X(:,2:end) * beta(2:end) + X(:,k) * beta(k)))/(X(:,k)' * W * X(:,k));
        lambda_k=lambda/(X(:,k)' * W * X(:,k));
        if abs(C(k)) >= 3/4*(lambda_k^(2/3))
            phi = acos(lambda_k/8*((abs(C(k))/3)^(-1.5)));
            beta(k) = 2/3*C(k)*(1 + cos(2/3*(pi - phi)));
        else
            beta(k) = 0;
        end
    end
    beta(1)=beta_zero;
    u = exp(X * beta)./(1 + exp(X * beta));
    W = diag(u .* (1 - u));
    z = X * beta + pinv(W) * (Y - u);
    iter=iter+1;
    toc(t_start);
end