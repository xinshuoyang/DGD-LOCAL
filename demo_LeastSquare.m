close all;rng(0)


%% Demo code of using DGD to solve least square problem

J = 10; % number of nodes in the network
m = 5;  % matrix size

% generate matrix A_j and b_j
A = zeros(m,m,J);
for j = 1:J
    [q,~] = qr(randn(m));
    A(:,:,j) = q*diag(rand(m,1))*q';
end
b = randn(m,J);

% optimization variables
X0 = 100*randn(m,J);

% weight matrix of the network
W = GenerateGraph(J, 0.3);

% stepsize
alpha = 1e-2;

% number of iterations
niter = 2000;

% errors
err_cost = zeros(niter,1);
err_cons = zeros(niter,1);

for iter = 1:niter
    X1 = zeros(m,J);
    
    for j = 1:J
        
        % consensus step
        for i = 1:J
            X1(:,j) = X1(:,j)+W(j,i)*X0(:,i);
        end
        
        % gradient step
        X1(:,j) = X1(:,j)-alpha*(A(:,:,j)*(X0(:,j)-b(:,j)));
        
        % compute approximation error
        err_cost(iter) = err_cost(iter)+0.5*((X1(:,j)-b(:,j))'*A(:,:,j)*(X1(:,j)-b(:,j)));
    end
    
    % compute consensus error
    X_mean = mean(X1, 2);
    for j = 1:J
        err_cons(iter) = err_cons(iter)+norm(X1(:, j)-X_mean,'fro')^2;
    end

    X0 = X1;    
end


% plots of approximation error and consensus error
figure()
fontsize = 18;
fig = semilogy(1:niter, err_cost, 1:niter, err_cons);
set(fig, {'LineStyle', 'LineWidth'}, {'-', 2; '--', 2})
h = legend('$\frac{1}{2}\sum_{j}\left(x^j-b_j\right)^T A_j \left( x^j-b_j\right)$', '$\sum_{j}\|x^j - \frac{1}{J}\sum_i x^i\|^2$');
set(h,'FontSize',30);
set(h,'Interpreter','latex');
xlabel('iteration','FontSize',fontsize,'FontName','Times New Roman');
set(gca, 'LineWidth' , 2 , 'FontSize', fontsize,'FontName'   , 'Times New Roman');
set(gcf,'position',[100 100 700 350])
set(gcf, 'Color', 'white');
% export_fig 'LS.pdf' -nocrop

