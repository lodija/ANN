function [threshold_value] = thresholdFunction(original_values, H)
%THRESHOLDFUNCTION calculates whether or not a value is 0 or 1 on based off of a
%high threshold value. This function is intended for rewriting output
%values for a neural network with a tanh/sigmoid range
%   for single-column inputs
threshold_value = zeros(size(original_values));
    for i = 1:length(original_values)
        if(original_values(i) >= H)
            threshold_value(i) = 1;
        elseif(original_values(i) < H)
            threshold_value(i) = 0;
        end
    end
end

