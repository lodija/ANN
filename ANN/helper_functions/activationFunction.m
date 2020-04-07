function activation = activationFunction(u)
%ACTIVATIONFUNCTION applies atanh(bu) to a value
    alpha = 1.7159; %Le Cun(1989,1993) suggests
    b = (2/3); %Le Cun(1989,1993) suggests
    
    %so that f(1) = 1, f(-1) = -1
    activation = alpha * tanh((b * u));
    
    
end

