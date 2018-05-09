function beta = Logistic_MCP_func(X,Y,beta_int,lambda)

    col=size(X,2);
    row=size(X,1);
% Step 1: Initialize (u,w,z) %
    u = exp(X * beta_int)./(1 + exp(X * beta_int));
    W = diag(u .* (1 - u));
    z = X * beta_int + pinv(W) * (Y - u);

% Step 2: The coordinate descent algorithm for sparse logistic with the Lasso penalty %
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
        
        gamma=19;%20;
        if abs(C(k)) <= gamma*lambda_k
            if C(k) > lambda_k
                beta(k) = (C(k) - lambda_k)/((1-1/gamma));
            elseif C(k) < -lambda_k
                beta(k) = (C(k) + lambda_k)/((1-1/gamma));
            elseif abs(C(k)) <= lambda_k
                beta(k) = 0;
            end
        else
            beta(k)=C(k);
        end
    end
    beta(1)=beta_zero;
    u = exp(X * beta)./(1 + exp(X * beta));
    W = diag(u .* (1 - u));
    z = X * beta + pinv(W) * (Y - u);
    iter=iter+1;
    toc(t_start);
end