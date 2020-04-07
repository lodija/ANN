%   This requires loading an the autoencoder created previously
%   is because there is no test or training set created for this
%   autoencoder. The data imported is the test and training sets from 
%   autoencoder

%% Run Training Data
clear; close; clc;
%   load data saved from autoencoder %
uiopen('load'); 
output_length = 784;    %output size = input size for autoencoder
input_length = 784;     % 28 x 28 input
number_of_requested_layers = 1; %used from parameters in part 1
number_of_requested_neurons = 200; %used from parameters in part 1
num_epochs = 250;
%   Used buildMatrix function from part 1
[weight_matrix, bias_matrix, activation_matrix, ~, ...
~, ~, last_weight_change_matrix, last_bias_change_matrix] = ... 
buildNetwork(input_length, output_length, number_of_requested_layers, number_of_requested_neurons);
current_epoch = 0;

%       train the network       %
training_ten_epoch_loss_vector = zeros(1,floor(num_epochs/10)); %vector for storing loss every 10 epochs
training_mean_loss = 1000;  %mean loss across all numbers
minibatch_size = 16; %size of minibatch
average_threshold = 5; %set threshold for error average for all numbers to be under
mean_loss = 15;
training_mean_loss_vector = average_threshold * ones(1,10); %create a vector for holding the mean loss for each number
ten_epoch_index = 1;

eta_o = 0.001; %learning rate of output
eta_h = 0.0001; %learning rate of input
beta = 1; %used for sparseness
lambda = 0.0001;
roh = 0.05; %sparseness target
roh_hat_matrix = cell(length(activation_matrix) -1);
epoch_loss = zeros(1,minibatch_size);

for b = 1:length(roh_hat_matrix)
    roh_hat_matrix{b} = 0 * activation_matrix{b};
end

empty_roh_hat_matrix = roh_hat_matrix;
M = minibatch_size;
tic
for i = 1:num_epochs  %run until there is a low error for ALL numbers
    % calculate p_j by running over entire training set without updating
    % weights
    %================== Shuffle the training set; make a subset===========%
    temp_training_set = training_set(randperm(size(training_set,1)),:); %randomize rows in the training set
    temp_training_set_labels = temp_training_set(:,end); %create label vector for training set
    temp_training_set(:,end) = [];                  %delete the label column from the grayscale matrix
    subset = temp_training_set(1:minibatch_size,:); %create a subset from the randomized training_set
    subset_labels = temp_training_set_labels(1:minibatch_size);
    
    %=============Calculate activation sums of hidden layers =============%
    roh_hat_matrix = empty_roh_hat_matrix; %clear the activation accumulator
    for j = 1:size(subset,1)    %for the entire training set
        input_vector = subset(j,:);
        %  calculate outputs     %
        for b = 1:size(activation_matrix,2) %for every layer
            for row = 1:size(activation_matrix{b},1) %for every neuron
                if(b == 1) %  calculate for input layer
                    activation_matrix{b}(row,1) = activationFunction...
                        (dot(weight_matrix{b}(row,:),input_vector(1,:)) + ...
                        bias_matrix{b}(row,1));
                else        %   calculate for anything else     %
                    activation_matrix{b}(row,1) = activationFunction... 
                    (dot(weight_matrix{b}(row,:), activation_matrix{b-1}(:,1)) + ...
                        bias_matrix{b}(row,1));
                end
            end
            if(b <= number_of_requested_layers) %only do this for the hidden layers; not the ouput layer
                roh_hat_matrix{b} = roh_hat_matrix{b} + activation_matrix{b};
            end
        end
    end
    %=======end activation accumulation==========%
    
    
    for b = 1:length(roh_hat_matrix)
            roh_hat_matrix{b} = roh_hat_matrix{b} / M; %divide all roh_hat changes by M (total number of pieces of data)
    end

    %============Begin learning with sparseness=============%
    for j = 1:size(subset,1)    %for the entire training set
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
        epoch_loss(j) = training_loss;
        
        %   store loss to specific number index         %
        output_label = subset_labels(j) + 1; %gives the value + 1 since the values are 0-9 and matlab is 1-based indexing (0 is at 1 in the vector)
        training_mean_loss_vector(output_label) = training_loss; %for every number, place the error on that number

        if(mean(training_mean_loss_vector) < average_threshold)
            break;
        end

        %       backpropagation using momentum       %
        [updated_weights, updated_biases, last_weight_change_matrix, ...
            last_bias_change_matrix] = backProp_sparse_weightDecay(weight_matrix, ...
            activation_matrix, bias_matrix, input_vector, y, y_hat, ...
            eta_o, eta_h, last_weight_change_matrix, last_bias_change_matrix, ...
            beta, roh, roh_hat_matrix);
        
        %       Accumulate weight and bias changes, store for later
        %       updating
        weight_matrix = updated_weights;
        bias_matrix = updated_biases;

    end
    
    if((mod(i,10) == 0 )|| i == 1)
       training_ten_epoch_loss_vector(ten_epoch_index) = training_loss;
       ten_epoch_index = ten_epoch_index  + 1;
    end
    
    %output during runtime so that you can see the network is actually
    %doing stuff:
    %sprintf reference: https://stackoverflow.com/questions/14924181/how-to-display-print-vector-in-matlab#27841544
    tlv = sprintf('%4.2f ', training_mean_loss_vector);
    
    fprintf('\nTraining Loss Vector: %s\n', tlv);
    mean_loss = mean(epoch_loss);
    fprintf('Average Loss = %2.2f\n', mean_loss);
    training_mean_loss = mean(training_mean_loss_vector);
    fprintf('Training Loss Mean: %2.2f\n', training_mean_loss);
    fprintf('Current epoch: %i\n\n', i);
    current_epoch = current_epoch + 1;
    if(mean(training_mean_loss_vector) < average_threshold)
            break;
    end
end
toc;

%% Run Test Data
test_loss = 1;
test_ten_epoch_loss_vector = []; %vector for storing loss every 10 epochs
test_total_loss_vector = zeros(size(test_set)); %loss over the past 5 minibatch runs
test_mean_loss_vector = 10 * ones(1,10); %create a vector for holding the mean loss for each number
test_mean_loss = mean(test_mean_loss_vector);
test_loss_sum = 0; %sum up all the loss

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
    test_loss_sum = test_loss_sum + test_loss;
    %   store loss to specific number index         %
    output_label = test_set_labels(j) + 1;
    test_mean_loss_vector(output_label) = test_loss; %for every number, place the error on that number
    test_total_loss_vector(j) = test_loss;    %store error for every epoch

end
average_loss_on_test_run = test_loss_sum / size(test_set,1);


%%      Plot images
% Code used for plotting images was adapted from code written by Dr. Ali Minai
figure('name', 'Input Weight Layer');
for current_epoch=1:14
    for j = 1:14
        v = reshape(weight_matrix{1}((current_epoch-1)*14+j,:),28,28);
        subplot(14,14,(current_epoch-1)*14+j)
        image(900 * v)
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
for current_epoch=1:14
    for j = 1:14
        v = reshape(b((current_epoch-1)*14+j,:),28,28);
        subplot(14,14,(current_epoch-1)*14+j)
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


%% Final plots to compare test and training runs
training_bar = mean(training_mean_loss_vector);
test_bar = mean(test_mean_loss_vector);
figure('name', 'Loss of Training and Test');
a = bar([1, 2],[training_bar, test_bar]);
hold on;
set(gca,'xticklabel', {'Training Loss Average', 'Test Loss Average'});
temp_vec = [training_bar, test_bar];
% https://www.mathworks.com/matlabcentral/answers/351875-how-to-plot-numbers-on-top-of-bar-graphs
text(1:length(temp_vec),temp_vec,num2str(temp_vec'),'vert','bottom','horiz','center'); 
title('Final Training and Test Loss');
figure('name', 'Loss at a Given Epoch');
plot(training_ten_epoch_loss_vector);
xlabel('Epoch');
ylabel('Loss');
title('Loss Every 10 Epochs');
ticks = [5, 10, 15, 20, 25, 30];
ax = {'50';'100';'150';'200';'250';'300'};
set(gca,'xtick',ticks); 
set(gca,'xticklabel',ax);


%make bar graph comparing loss across numbers between training and test
%sets: https://stackoverflow.com/questions/46580337/how-to-plot-bar-graph-to-compare-two-quantities-on-y-axis-at-specific-values-on
figure('name', 'Number Loss');
bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[training_mean_loss_vector; test_mean_loss_vector].');
xlabel('Number Class (0-9)');
ylabel('Loss');
title('Training and Test Loss Across All Classes');
legend('Training Loss', 'Test Loss');

