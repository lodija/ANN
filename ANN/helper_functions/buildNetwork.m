function [weight_matrix, bias_matrix, activation_matrix, granted_neurons_per_layer, granted_layers, granted_neurons, last_weight_matrix_values, last_bias_matrix_values] = buildNetwork(input_length, output_length, number_of_requested_layers, number_of_requested_neurons)
%BUILDNETWORK builds a neural network of the desired number of hidden
%layers, and desired number of neruons. This function returns a weight
%matrix and a bias matrix. Additionally, it will return the number of
%hidden layers and neurons granted.
%   This function will set to make every hidden layer have the same number 
%   of neurons. The network will always have the number of requested layers
%   but the number of neurons in each layer will change.
%   
%   For instance, if you request 10 neurons and 3 hidden layers:
%
%   10 neurons, 3 layers    =>  3 layers, 9 neurons
%           x   x   x       =>      x   x   x
%           x   x   x       =>      x   x   x
%           x   x   x       =>      x   x   x
%           x   NaN NaN     =>      
%
%   If there is one layer, it will have as many neurons in one layer as is 
%   specified. 
%   
%   The first input is the input length of the input vector. The second
%   input is the length of the output vector. Third, is the number of
%   hidden layers requested, and fourth are the number of hidden neurons
%   requested. ANY NON-INTEGER VALUE WILL BE ROUNDED DOWN AT THE START OF 
%   THE NETWORK-BUILDING PROCESS
%   This function allows a return of empty weights and bias cells so that
%   they can be used as initial inputs for backpropagation with momentum
    clc;
    
    fprintf('Building network...\n');
    if (number_of_requested_layers < 0)    %negative numbers
        e = MException('MyComponent:noSuchVariable', ...
            'Number of hidden layers can''t be negative!\n');
        throw(e)
    elseif (number_of_requested_neurons < 0)   %negative numbers
        e = MException('MyComponent:noSuchVariable', ...
            'Number of neurons can''t be negative!\n');
        throw(e)
    elseif (number_of_requested_neurons < number_of_requested_layers) %if there are less desired neurons than desired hidden layers
        e = MException('MyComponent:noSuchVariable', ...
            'There cannot be more hidden layers than there are neurons!\n');
        throw(e)
    end

    %       Get the inputs to be whole numbers          %
    input_length = floor(input_length);
    output_length = floor(output_length);
    number_of_requested_layers = floor(number_of_requested_layers);
    number_of_requested_neurons = floor(number_of_requested_neurons);
    
    %       Get number of hidden layers to be made      %
    granted_neurons_per_layer = floor(number_of_requested_neurons/number_of_requested_layers);
    if(number_of_requested_layers == 0)
        granted_neurons_per_layer = 0;
    end
    granted_layers = number_of_requested_layers;
    
    %       Get granted neurons                         %
    granted_neurons = granted_neurons_per_layer * granted_layers;
    
    %       Build matrices                              %
    %create range for which weights to be initialized
    a = sqrt(6/(input_length + output_length)); % Xavier initialization (Golorot & Bengio, 2010)
%     a = sqrt(3/input_length);
    lower_bound = a;
    upper_bound = -a;
    
    %      Build cell matrix for holding biases of the network--also build
    %      the matrix for holding all of the activation values for all
    %      neurons
    bias_matrix = cell(1, granted_layers + 1); %contains at minimum, one layer of baises (output layer)
    last_bias_matrix_values = cell(1, granted_layers + 1);
    activation_matrix = bias_matrix; %activation matrix has the same setup as the bias matrix (one per neuron)
    for i = 1:length(bias_matrix)
        if(granted_layers == 0 || i == length(bias_matrix)) %if there are no hidden layers (only input and output) || end of matrix is met
            %https://www.mathworks.com/help/matlab/math/floating-point-numbers-within-specific-range.html
            %weight initialization between +a and -a
            bias_matrix{i} = (upper_bound - lower_bound).*rand(output_length, 1) + lower_bound; %randomize bias for layer
            activation_matrix{i} = zeros(output_length,1); %make output layer activation
            last_bias_matrix_values{i} = zeros(output_length, 1);
            break; %matrix is done
        end
        bias_matrix{i} = (upper_bound - lower_bound).*rand(granted_neurons_per_layer, 1) + lower_bound; %create random biases from -1 to 1
        activation_matrix{i} = zeros(granted_neurons_per_layer,1);
        last_bias_matrix_values{i} = zeros(granted_neurons_per_layer, 1);
    end
    
    %       Build cell matrix of matrices that hold the weights of each
    %       layer.
    weight_matrix = cell(1, granted_layers + 1); %create cell array to hold each layer of weights
    last_weight_matrix_values = cell(1, granted_layers + 1);
    
    for i = 1:length(weight_matrix) %calculate the number of weights needed per layer, and create a matrix for those weights at each layer
        %       single layer network (input and output)        %
        if(granted_layers == 0) %if there are no hidden layers (only input and output)
            weight_matrix{i} = (upper_bound - lower_bound).*rand(output_length, input_length) + lower_bound; %input mapped to output
            last_weight_matrix_values{i} = zeros(output_length, input_length);
            break; %matrix is done
        end
       
        %       multi-layer network         %
        if(i == 1)                          %input weight layer
            weight_matrix{i} = (upper_bound - lower_bound).*rand(granted_neurons_per_layer, input_length) + lower_bound; %input mapped onto hidden layer
            last_weight_matrix_values{i} = zeros(granted_neurons_per_layer, input_length);
        elseif(i == length(weight_matrix))  %output weight layer
            weight_matrix{i} = (upper_bound - lower_bound).*rand(output_length, granted_neurons_per_layer) + lower_bound; %hidden layer mapped to output
            last_weight_matrix_values{i} = zeros(output_length, granted_neurons_per_layer);
        else                                %hidden weight layers
            weight_matrix{i} = (upper_bound - lower_bound).*rand(granted_neurons_per_layer, granted_neurons_per_layer) + lower_bound; %hidden layer mapped on hidden layer
            last_weight_matrix_values{i} = zeros(granted_neurons_per_layer, granted_neurons_per_layer);
        end
    end
    fprintf('Matrix built!');
end

