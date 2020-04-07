function total_loss = lossFunction(y, y_hat)
%LOSSFUNCTION calculates the overall loss of the system
% the loss is defined as 0.5 * sum(y-y_hat)^2

total_loss = 0.5 * sum(y-y_hat).^2;
end

