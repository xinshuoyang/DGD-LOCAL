close all;rng(0)

%% Demo code of using DGD+LOCAL to solve low-rank matrix factorization problem

J = 10; % number of nodes in the network
n = 50; mj = 20; m = mj*J; % dimensions of data matrix
r = 10; % rank of the data matrix

Yt = randn(n,r)*randn(r,m);
Y = reshape(Yt,n,mj,J);

% optimization variables and initialization

U = randn(n,r,J);V = randn(mj,r,J);
U1 = zeros(n,r,J);V1 = zeros(mj,r,J);

% network weight matrix
W = GenerateGraph(J, 0.3);

% stepsize
lambda = 1e-3;

% number of iterations
niter = 3000;

% errors 
err_cost = zeros(niter,1);
err_cons = zeros(niter,1);
  

for iter = 1:niter
    
    for j = 1:J
        % gradient and consensus step for variable U
        U1(:,:,j) = -2*lambda*(U(:,:,j)*V(:,:,j)'-Y(:,:,j))*V(:,:,j);
        for i = 1:J
            U1(:,:,j) = U1(:,:,j)+W(j,i)*U(:,:,i);
        end
        
        % gradient step for variable V
        V1(:,:,j) = V(:,:,j)-2*lambda*(U(:,:,j)*V(:,:,j)'-Y(:,:,j))'*U(:,:,j);
        
        % compute objective value
        err_cost(iter) = err_cost(iter)+norm(U1(:,:,j)*V1(:,:,j)'-Y(:,:,j),'fro')^2;
    end
    
    % concensus error
    U_mean = mean(U1, 3);
    for j = 1:J
        err_cons(iter) = err_cons(iter)+norm(U1(:, :, j)-U_mean,'fro')^2;
    end
    
    U = U1;
    V = V1;
end


% plots of approximation error and consensus error
figure()
fontsize = 18;
fig = semilogy(1:niter, err_cost, 1:niter, err_cons);
set(fig, {'LineStyle', 'LineWidth'}, {'-', 2; '--', 2})
h = legend('$\sum_{j}\|U^j V_j^T - Y_j\|_F^2$', '$\sum_{j}\|U^j - \frac{1}{J}\sum_i U^i\|_F^2$');
xlabel('iteration','FontSize',fontsize,'FontName','Times New Roman');
set(h,'FontSize',30);
set(h,'Interpreter','latex');
set(gca, 'LineWidth' , 2 , 'FontSize', fontsize,'FontName'   , 'Times New Roman');
set(gcf,'position',[100 100 700 350])
set(gcf, 'Color', 'white');
% export_fig 'MF.pdf' -nocrop

