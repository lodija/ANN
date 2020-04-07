%   It is required that you run an ANN before running this 
%   code. This is because there is no test or training set created for this
%   autoencoder. The data imported is the test and training sets from 
%   the ANN script
%   You could get around the requirement by using the "loadSet()" function
%   to create a test and training set from the MNIST dataset. 
%   
%  
%   Part 2: Create an autoencoder, use the same test and training data from
%   problem 1, as well as the same size of hidden neurons and hidden layers

clear; close; clc;

%   load data saved from part 1 (test and training set)     %   
uiopen('load');

output_length = 784;    %output size = input size for autoencoder
input_length = 784;     % 28 x 28 input
number_of_requested_layers = 1; %used from parameters in part 1
number_of_requested_neurons = 200; %used from parameters in part 1

%   Used buildMatrix function from part 1
[weight_matrix, bias_matrix, activation_matrix, ~, ...
~, ~, last_weight_change_matrix, last_bias_change_matrix] = ... 
buildNetwork(input_length, output_length, number_of_requested_layers, number_of_requested_neurons);
i = 0;
%       train the network       %
eta_o = 0.001; %learning rate of output
eta_h = 0.0001; %learning rate of input
minibatch_size = 1000; %size of minibatch
training_loss = 1;
training_ten_epoch_loss_vector = []; %vector for storing loss every 10 epochs
training_total_loss_vector = []; %loss over the past 5 minibatch runs
training_mean_loss_vector = 10 * ones(1,10); %create a vector for holding the mean loss for each number
training_mean_loss = 20;  %mean loss across all numbers
average_threshold = 5; %set threshold for error average for all numbers to be under
while(training_mean_loss > average_threshold)  %run until there is a low error for ALL numbers
    tic;
    training_mean_loss = mean(training_mean_loss_vector);
    pass_count = mod(i,10); %goes to zero every 10th epoch
    temp_training_set = training_set(randperm(size(training_set,1)),:); %randomize rows in the training set
    temp_training_set_labels = temp_training_set(:,end); %create label vector for training set
    temp_training_set(:,end) = [];                  %delete the label column from the grayscale matrix
    subset = temp_training_set(1:minibatch_size,:); %create a subset from the randomized training_set
    subset_labels = temp_training_set_labels(1:minibatch_size);
    %   for the minibatch:      %
    for j = 1:size(subset,1)    %for the minibatch
        input_vector = subset(j,:);
        %  calculate outputs     %
        for b = 1:size(activation_matrix,2) %for every layer
            for row = 1:size(activation_matrix{b},1) %for every neuron
                if(b == 1) %  calculate for input layer
                    activation_matrix{b}(row,1) = activationFunction(dot ...
                        (weight_matrix{b}(row,:),input_vector(1,:)) + ...
                        bias_matrix{b}(row,1));
                else        %   calculate for anything else     %
                    activation_matrix{b}(row,1) = activationFunction(dot ...
                        (weight_matrix{b}(row,:), activation_matrix{b-1}(:,1)) + ...
                        bias_matrix{b}(row,1));
                end
            end
        end
        %   calculate loss      %
        y = input_vector';
        y_hat = activation_matrix{end}; %actual output
        training_loss = lossFunction(y, y_hat);
        
        %   store loss to specific number index         %
        output_label = subset_labels(j) + 1; %gives the value + 1 since the values are 0-9 and matlab is 1-based indexing (0 is at 1 in the vector)
        training_mean_loss_vector(output_label) = training_loss; %for every number, place the error on that number
        if(mean(training_mean_loss_vector) < average_threshold) %learning rate across all is low enough
            break;
        end
        %       backpropagation using momentum       %
        [updated_weights, updated_biases, last_weight_change_matrix, ...
            last_bias_change_matrix] = backProp(weight_matrix, ...
            activation_matrix, bias_matrix, input_vector, y, y_hat, ...
            eta_o, eta_h, last_weight_change_matrix, last_bias_change_matrix);
        weight_matrix = updated_weights;
        bias_matrix = updated_biases;
        
    end
    if(~pass_count || i == 1) %after first epoch, or every 10
        training_ten_epoch_loss_vector = [training_ten_epoch_loss_vector ; training_loss];    %store error every 10 epoch for plotting
    end
    training_total_loss_vector = [training_total_loss_vector ; training_loss];    %store error for every epoch
    
    %output during runtime so that you can see the network is actually
    %doing stuff:
    %sprintf reference: https://stackoverflow.com/questions/14924181/how-to-display-print-vector-in-matlab#27841544
    tlv = sprintf('%2.2f ', training_mean_loss_vector);
    fprintf('\nTraining Loss Vector: %s\n', tlv);
    if(size(training_total_loss_vector, 1) > 6) 
        mean_loss = mean(training_total_loss_vector((end - 5):end),1);
        fprintf('Average Loss = %2.2f\n', mean_loss);
    end
    fprintf('Training Loss Mean: %2.2f\n', training_mean_loss);
    i = i + 1;
    toc;
end



%% Run Test Data

test_loss = 1;
test_ten_epoch_loss_vector = []; %vector for storing loss every 10 epochs
test_total_loss_vector = zeros(size(test_set)); %loss over the past 5 minibatch runs
test_mean_loss_vector = 10 * ones(1,10); %create a vector for holding the mean loss for each number
test_mean_loss = mean(test_mean_loss_vector);
pass_count = mod(i,10); %goes to zero every 10th epoch
TP = 0;
temp_test_set = test_set(randperm(size(test_set,1)),:); %randomize rows in the training set
temp_test_set_labels = temp_test_set(:,end); %create label vector for training set
temp_test_set(:,end) = [];                  %delete the label column from the grayscale matrix
subset = temp_test_set(1:minibatch_size,:); %create a subset from the randomized test_set
subset_labels = temp_test_set_labels(1:minibatch_size);
%   for the minibatch:      %
for j = 1:size(test_set,1)    %for the minibatch
    input_vector = test_set(j,:);
    %  calculate outputs     %
    for b = 1:size(activation_matrix,2) %for every layer
        for row = 1:size(activation_matrix{b},1) %for every neuron
            if(b == 1) %  calculate for input layer
                activation_matrix{b}(row,1) = activationFunction(dot ...
                    (weight_matrix{b}(row,:),input_vector(1,:)) + ...
                    bias_matrix{b}(row,1));
            else        %   calculate for anything else     %
                activation_matrix{b}(row,1) = activationFunction(dot ...
                    (weight_matrix{b}(row,:), activation_matrix{b-1}(:,1)) + ...
                    bias_matrix{b}(row,1));
            end
        end
    end
    %   calculate loss      %
    y = input_vector';
    y_hat = activation_matrix{end}; %actual output
    test_loss = lossFunction(y, y_hat);

    %   store loss to specific number index         %
    output_label = test_set_labels(j) + 1;
    test_mean_loss_vector(output_label) = test_loss; %for every number, place the error on that number
    test_total_loss_vector(j) = test_loss;    %store error for every epoch

end
if(~pass_count || i == 1) %after first epoch, or every 10
    test_ten_epoch_loss_vector = [test_ten_epoch_loss_vector ; test_loss];    %store error every 10 epoch for plotting
end

%output during runtime so that you can see the network is actually
%doing stuff:
%sprintf reference: https://stackoverflow.com/questions/14924181/how-to-display-print-vector-in-matlab#27841544
tlv = sprintf('%2.2f ', test_mean_loss_vector);
fprintf('\nTest Loss Vector: %s\n', tlv);
if(size(test_total_loss_vector, 1) > 6) 
    mean_loss = mean(test_total_loss_vector((end - 5):end),1);
    fprintf('Average Loss = %2.2f\n', mean_loss);
end
fprintf('Test Loss Mean: %2.2f\n', test_mean_loss);


%% Final plots to compare test and training runs
figure('name', 'Loss of Training and Test');
a = bar([1, 2],[training_total_loss_vector(end), test_total_loss_vector(end,1)]);
hold on;
%bar(2,, 'r');
set(gca,'xticklabel', {'Training Loss', 'Test Loss'});
temp_vec = [training_total_loss_vector(end), test_total_loss_vector(end,1)];
% https://www.mathworks.com/matlabcentral/answers/351875-how-to-plot-numbers-on-top-of-bar-graphs
text(1:length(temp_vec),temp_vec,num2str(temp_vec'),'vert','bottom','horiz','center'); 
title('Final Training and Test Loss');
figure('name', 'Loss at a Given Epoch');
plot(training_ten_epoch_loss_vector);
xlabel('Epoch');
ylabel('Loss');
title('Loss Every 10 Epochs');
%make bar graph comparing loss across numbers between training and test
%sets: https://stackoverflow.com/questions/46580337/how-to-plot-bar-graph-to-compare-two-quantities-on-y-axis-at-specific-values-on
figure('name', 'Number Loss');
bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[training_mean_loss_vector; test_mean_loss_vector].');
xlabel('Number Class (0-9)');
ylabel('Loss');
title('Training and Test Loss Across All Classes');
legend('Training Loss', 'Test Loss');

%%      Plot images
% Code used for plotting images was adapted from code written by Dr. Ali Minai
figure('name', 'Input Weight Layer');
for i=1:14
    for j = 1:14
        v = reshape(weight_matrix{1}((i-1)*14+j,:),28,28);
        subplot(14,14,(i-1)*14+j)
        image(1 * v)
        colormap(gray(64));
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
        set(gca,'dataaspectratio',[1 1 1]);
    end
end

figure('name','Output Weight layer');
b = weight_matrix{2}';
for i=1:14
    for j = 1:14
        v = reshape(b((i-1)*14+j,:),28,28);
        subplot(14,14,(i-1)*14+j)
        image(900 * v)
        colormap(gray(64));
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
        set(gca,'dataaspectratio',[1 1 1]);
    end
end

figure('name', 'Lowest Error Number Reconstruction');
c = y_hat';
v = reshape(c((1-1)*1+1,:),28,28);
image(64 * v)
colormap(gray(64));
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
set(gca,'dataaspectratio',[1 1 1]);

figure('name', 'Actual Number');
d = y';
v = reshape(d((1-1)*1+1,:),28,28);
image(64 * v)
colormap(gray(64));
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
set(gca,'dataaspectratio',[1 1 1]);

