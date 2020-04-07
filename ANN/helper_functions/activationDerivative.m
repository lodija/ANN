function deriv = activationDerivative(u)
%ACTIVATIONDERIVATIVE is the derivative of the activation function:
% alpha = 1.7159; %Le Cun(1989,1993) suggests
%     b = (2/3); %Le Cun(1989,1993) suggests
%     
%     %so that f(1) = 1, f(-1) = -1
%     activation = alpha * tanh((b * u));
%  This function is used to speed up calculation speed of neural network
%  (so that the derivative of the function doesn't have to constantly be
%  calculated. Derivative was calculated on Wolfram Alpha
%  (http://www.wolframalpha.com/)
    alpha = 1.14393;
    b = 2/3;
    deriv = alpha * sech(b * u) .^ 2;
end

