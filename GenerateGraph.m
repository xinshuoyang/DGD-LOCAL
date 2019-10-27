function [networkGraph] = GenerateGraph(N, varargin)
    p = varargin{1};
    
    %Erdos-Reyni random graphs
    A = rand(N,N);
    A = triu(A,1);
    A = A + A';
    G = @(p) A < p;
    
    networkGraph = double(G(p));
    for k=1:N
        networkGraph(k,k)=0;
    end
    temp = networkGraph;
    for k=1:N
        for j=1:N
            networkGraph(k, j) = 1/max(sum(temp(k, :)), sum(temp(j, :)))*temp(k, j);
        end
    end
    for k=1:N
        networkGraph(k,k)= 1-sum(networkGraph(k, :));
    end
    
    if find(isnan(networkGraph))
        networkGraph = GenerateGraph(N, p);
    end
end

