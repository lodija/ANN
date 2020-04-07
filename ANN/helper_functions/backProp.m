function [updated_weight_layers, updated_bias_layers, last_weight_change_layers, last_bias_change_layers] = backProp(weight_layers, activation_layers, bias_layers, input, y, y_hat, eta_o, eta_h, last_weight_change_layers, last_bias_change_layers)
%BACKPROP takes in a neural network's weight layer, bias layer, input
%layer, and activations. It returns the updated weights and biases. This
%function is to be used with the "buildNetwork.m" function
%   all layers other than the inputs are assumed to be cell vectors
%   containing matrices of the weights and baises. This updates weights and
%   biases with momentum

    alpha = 0.9; %for momentum (set to 0 if you want no momentum)
    updated_weight_layers = cell(size(weight_layers));
    updated_bias_layers = cell(size(bias_layers));
    delta_layers = cell(size(activation_layers));
    
    end_of_network  = length(activation_layers);
    %       Backpropagate the error         %
    for i = end_of_network:-1:1 %starting from the back...
            if(i == end_of_network) %  Should be done first (output layer)
                delta_layers{i} = activationDerivative(activation_layers{i}) .* (y-y_hat);
            else        %   calculate for every other layer     %
                delta_layers{i} = activationDerivative(activation_layers{i}) * sum(weight_layers{i+1}' * delta_layers{i+1});
            end
    end
    
    %       Update the weights based off the error      %
    for i = 1:end_of_network %update everything         
            if(i == 1)        %   calculate for input layer         %
                temp_w_change = (eta_h * delta_layers{i} * input); %save computations
                temp_b_change = (eta_h * delta_layers{i}); %save computations
                updated_weight_layers{i} = weight_layers{i} + temp_w_change + (alpha * last_weight_change_layers{i});
                updated_bias_layers{i} = bias_layers{i} + temp_b_change + (alpha * last_bias_change_layers{i});
                last_weight_change_layers{i} = temp_w_change;
                last_bias_change_layers{i} = temp_b_change;
            elseif(i == end_of_network) %  Should be done for the output layer
                %use momentum to update weights 
                temp_w_change = (eta_o * delta_layers{i} * activation_layers{i-1}'); %save computations
                temp_b_change = (eta_o * delta_layers{i}); %save computations
                updated_weight_layers{i} = weight_layers{i} + temp_w_change + (alpha * last_weight_change_layers{i}); %update weights
                updated_bias_layers{i} = bias_layers{i} + temp_b_change + (alpha * last_bias_change_layers{i}); %input layer is all 1 for bias
                last_weight_change_layers{i} = temp_w_change;
                last_bias_change_layers{i} = temp_b_change;
            else
                temp_w_change = (eta_h * delta_layers{i} * activation_layers{i-1}'); %save computations
                temp_b_change = eta_h * delta_layers{i}; %save computations
                updated_weight_layers{i} = weight_layers{i} + temp_w_change + (alpha * last_weight_change_layers{i});
                updated_bias_layers{i} = bias_layers{i} + temp_b_change + (alpha * last_bias_change_layers{i});
                last_weight_change_layers{i} = temp_w_change;
                last_bias_change_layers{i} = temp_b_change;
            end
    end
end

