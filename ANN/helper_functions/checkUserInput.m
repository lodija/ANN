function [ is_valid_input ] = checkUserInput(num_hidden, num_neurons, out_learn, hidden_learn)
%CHECKUSERINPUT is a function made for checking whether or not the user
%input is valid. This function is
%separate just to keep the code cleaner. It returns 0 (invalid input) or 1
%(valid input).
%   Checks the number of hidden layers and neurons to make sure that
%   numbers input won't break the network

    if (num_hidden < 0 || num_neurons < 0 || out_learn < 0 || hidden_learn < 0)
        is_valid_input = 0;
    elseif (num_neurons < num_hidden) %if there are less desired neurons than desired hidden layers
        is_valid_input = 0;
    else
        is_valid_input = 1;
    end


end

