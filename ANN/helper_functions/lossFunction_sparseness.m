function total_loss = lossFunction_sparseness(y, y_hat, beta, kl_divergence)
%LOSSFUNCTION calculates the overall loss of the system. Additionally,
%there is a penalty term added for implementing Sparseness
% the loss with sparseness is defined as [0.5 * sum(y-y_hat)^2] + 
% p*log(p/p_j) + (1-p)log((1-p)/(1-p_j)) => 
% loss = loss_function + kl_divergence
% (Kullback-Leibler (KL)divergence)
kl_divergence_sum = 0;
for i = 1:length(kl_divergence)
    
    kl_divergence_sum = kl_divergence_sum + sum(kl_divergence{i});
end
total_loss = 0.5 * sum(y-y_hat).^2 + beta .* kl_divergence_sum;
end

