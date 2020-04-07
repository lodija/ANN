function kl_divergence = average_activation(input_size, activation_layers)
%AVERAGE_ACTIVATION is a function used to calculate roh_j (p_j) for
%implementing sparseness by using the Kullback-Leibler Divergence
%   kl divergence is calculated using desired average activation value, roh (p), and
%   the average activation, roh_j (p_j)
    kl_divergence = (1/input_size) * sum(activation_layers);
end

