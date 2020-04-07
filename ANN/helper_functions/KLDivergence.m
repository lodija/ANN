function kl_divergence = KLDivergence(roh, roh_hat_matrix)
%AVERAGE_ACTIVATION is a function used to calculate roh_j (p_j) for
%implementing sparseness by using the Kullback-Leibler Divergence
%   kl divergence is calculated using desired average activation value, roh (p), and
%   the average activation, roh_j (p_j)
kl_divergence = roh_hat_matrix; %quick and dirty to get the right size cell array
    for i = 1:length(roh_hat_matrix)
        kl_divergence{i} = roh * log(roh / roh_hat_matrix{i}) + (1 - roh) * log((1 - roh)/(1 - roh_hat_matrix{i}));
    end
    
end


