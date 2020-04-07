
%% ANN
% Part 1:   Write a program implementing multi-layer feed-forward neural
% networks and train them using back-propegation and momentum. 
% good learning rates for this: 
%
%   1 layer
%   250 Neurons
%   Learning rate hidden neurons: 0.00001
%   Learning rate output: 0.001

close all; clc; clear; % housekeeping
                          
%% Get get the hyperparameters
% At the end of the network running, you're able to load an image of the
% network you were working on previously. This way you can pick up training
% where you left off
load_file = input('Load .mat file? y/n:\n','s');
can_load = strncmp('y', load_file, 1);
if(can_load)
    clear;
    uiopen('load');
    can_load = 1;
else
    num_epochs = input('Enter the number of epochs to run\n');
    eta_o = input('Enter learning rate for output\n'); %learning rate of output
    eta_h = input('Enter learning rate for hidden layers\n');

    number_of_hidden_layers = input('Enter the desired number of hidden layers:\n');
    number_of_hidden_neurons = input('Enter the number of desired hidden neurons:\n');
    H = 0.75 * activationFunction(inf); % '1' target value for activation function
    L = 0.75 * activationFunction(-inf); % '0' target value for activation function
    activation_max = activationFunction(inf);
    activation_min = activationFunction(-inf);
    
    %       Make sure user inputs are valid                 
    is_valid = checkUserInput(number_of_hidden_layers, number_of_hidden_neurons, eta_o, eta_h);
    if (~is_valid)
        e = MException('MyComponent:noSuchVariable', ...
            'User inputs must be valid!\n',is_valid);
        throw(e)
    end
    
    %%      Prepare Data for passing to network     
    % Since the data needs to be randomized, I included the labels vector with the
    % the grayscale values so that they could be shuffled simultaneously and
    % not lose their respective order
  
    clc;
    fprintf('Loading data...\n');
    data_set = load('MNISTnumImages5000.txt');      %import dataset
    data_labels = load('MNISTnumLabels5000.txt');   %import label to dataset
    data_set = [data_set data_labels];    %concatinate dataset and labels
    
    split_ratio  = 0.8; %80 percent of the data will belong to the training set
    [training_set, test_set] = loadSet(data_set, split_ratio); % shuffle rows and create training and test set 
    temp_training_set = zeros(size(training_set));
    training_set_labels = training_set(:,end);
    data_set(:,end) = [];
    test_set_labels = test_set(:,end);         %create label vector for test set 
    test_set(:,end) = [];                      %delete label column from grayscale matrix
    
    output_layer = zeros(10,1); % create 10 x 1 output layer because there are 10 total numbers
    output_length = 10; % there will be 10 possible outputs
    input_length = size(test_set, 2);
    
    %%      Build the network          
   [weight_matrix, bias_matrix, activation_matrix, granted_neurons_per_layer, ...
   granted_layers, granted_neurons, last_weight_change_matrix, ...
   last_bias_change_matrix] = buildNetwork(input_length, output_length, ...
    number_of_hidden_layers, number_of_hidden_neurons);

    fprintf('\n\nRequested Hidden Layers: %i \n', number_of_hidden_layers);
    fprintf('Granted Hidden Layers: %i \n', granted_layers);
    fprintf('\nGranted Hidden Neurons Per Layer: %i \n', granted_neurons_per_layer);
    fprintf('Total Granted Hidden Neurons: %i \n\n', granted_neurons);

    %%      Prepare Build Statistics           
    num_epochs = floor(num_epochs); 
    minibatch_size = 1000; %size of the subset to be shown each epoch
    hit_rate = 0; %save this into hit-rate vector after every 10 epochs
    num_samples = floor(num_epochs/10) + 1;
    %initialize vectors for updating performance information

    TP_vector = zeros(1,num_samples); 
    vector_index = 1;
    y_hat = zeros(size(activation_matrix{end})); %create output expected
    error_vector = zeros(1, num_samples);
    hit_rate_vector = zeros(1, num_samples); % one value for the first run, and every 10 epochs after
        
end

%%      Train the network
training_confusion_matrix = zeros(output_length + 1,output_length + 1);
answers = zeros(minibatch_size,2);
fprintf('Training....\n');

for i = 1:num_epochs
    tic;
    minibatch_ran = 0;
    while(minibatch_ran < size(training_set,1))  
        TP = 0;
        temp_training_set = training_set(randperm(size(training_set,1)),:); %randomize rows in the training set
        temp_training_set_labels = temp_training_set(:,end); %create label vector for training set
        temp_training_set(:,end) = [];                  %delete the label column from the grayscale matrix
        subset = temp_training_set(1:minibatch_size,:); %create a subset from the randomized training_set
        subset_labels = temp_training_set_labels(1:minibatch_size);
        %   for the minibatch:      
        for j = 1:size(subset,1)    %for the minibatch
            y = zeros(size(y_hat));
            input_vector = subset(j,:);
            %  calculate outputs     
            for b = 1:size(activation_matrix,2) %for every layer
                for row = 1:size(activation_matrix{b},1) %for every neuron
                    if(b == 1) %  calculate for input layer
                        activation_matrix{b}(row,1) = activationFunction(dot ...
                            (weight_matrix{b}(row,:),input_vector(1,:)) + ...
                            bias_matrix{b}(row,1));
                    else        %   calculate for anything else     
                        activation_matrix{b}(row,1) = activationFunction(dot ...
                            (weight_matrix{b}(row,:), activation_matrix{b-1}(:,1)) + ...
                            bias_matrix{b}(row,1));
                    end
                end
            end
            y_hat = activation_matrix{end}; %actual output
            %       Calculate weights and biases using momentum. Store them for later updating; after minibatch is done        %
            y((subset_labels(j)+1)) = activation_max; %set corresponding y_hat value to 1 based off the label from the subset
            y = (y == 0) * activation_min + y; %vector math to get y_hat values = to their threshold allowances
            [TP_minibatch, training_confusion_matrix]  = performanceFunction(y, y_hat, training_confusion_matrix);
            TP = TP  + TP_minibatch;
            [~, indxy] = max(y);
            [~, indxy_hat] = max(y_hat);
            answers(j,1) = indxy;
            answers(j,2) = indxy_hat;
            
            %       backpropagation using momentum       
            [updated_weights, updated_biases, last_weight_change_matrix, ...
                last_bias_change_matrix] = backProp(weight_matrix, ...
                activation_matrix, bias_matrix, input_vector, y, y_hat, ...
                eta_o, eta_h, last_weight_change_matrix, last_bias_change_matrix);
            weight_matrix = updated_weights;
            bias_matrix = updated_biases;

        end
        minibatch_ran = minibatch_ran + minibatch_size; %each epoch consists of running the entire training set
        hit_rate = TP/minibatch_size;
        %output during runtime so that you can see the network is actually
        %doing stuff:
        fprintf('\n');
        fprintf('\nHit Rate: %i %%\n', floor(hit_rate*100));
        fprintf('Error = %f\n', sum(0.5 * (y-y_hat).^2));
        fprintf('Epoch: %i of %i\n', i, num_epochs);
    end
    
    TP_vector(vector_index) = TP;
    %       Calculate Hit Rate for every 10 epochs      
    hit_rate = TP/minibatch_size;
    hit_rate_vector(vector_index) = hit_rate;
    error_vector(vector_index) = 1-hit_rate;
    vector_index = vector_index + 1; %increase index for the next update
    fprintf('\nTraining Hit Rate: %i %%\n', floor(hit_rate*100));
    toc;
end


%%      Test the network
TP_test = 0;
TP_test_vector = zeros(size(TP_vector));
test_confusion_matrix = zeros(output_length + 1, output_length + 1);
%       Test set        % 
for i = 1:size(test_set,1)
   y = zeros(size(y_hat));
   input_vector = test_set(i,:);
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
    y_hat = activation_matrix{end}; %actual output
    y(test_set_labels(i)+1)= activation_max; %set corresponding y_hat value to 1 based off the label from the subset
    y = (y == 0) * activation_min + y; %vector math to get y_hat values = to their threshold allowances
    [TP_test_set, test_confusion_matrix] = performanceFunction(y, y_hat, test_confusion_matrix);
    TP_test = TP_test  + TP_test_set;
        
end

%%      Display network performance
hit_rate_test = TP_test/size(test_set,1);
fprintf('\nTest Hit Rate: %i %%\n', floor(hit_rate_test*100));
column_names = {'Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine','Total'};
row_names = column_names;

test_confusion_matrix(end,:) = sum(test_confusion_matrix,1);
test_confusion_matrix(:,end) = sum(test_confusion_matrix,2);

training_confusion_matrix(end,:) = sum(training_confusion_matrix,1);
training_confusion_matrix(:,end) = sum(training_confusion_matrix,2);

test_confusion_matrix_table = array2table(test_confusion_matrix,'VariableNames',column_names, 'RowNames', row_names);
training_confusion_matrix_table = array2table(training_confusion_matrix,'VariableNames',column_names, 'RowNames', row_names);

% create graphical table to view info:
% https://www.mathworks.com/matlabcentral/answers/254690-how-can-i-display-a-matlab-table-in-a-figure
figure('name', 'Confusion Matrix for Training Data');
training_confusion_matrix_graphics = uitable('Data',training_confusion_matrix_table{:,:},'ColumnName',training_confusion_matrix_table.Properties.VariableNames,...
    'RowName',training_confusion_matrix_table.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);
figure('name', 'Confusion Matrix for Test Data');
test_confusion_matrix_graphics = uitable('Data',test_confusion_matrix_table{:,:},'ColumnName',test_confusion_matrix_table.Properties.VariableNames,...
    'RowName',test_confusion_matrix_table.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);

figure('name', 'Error Rate and Epoch');
percent_error_vector = error_vector .* 100;
plot(linspace(1,100,length(percent_error_vector)), percent_error_vector);
xlabel('Epoch');
ylabel('Error');
title('Error Rate vs. Epoch');

% Offer user to save data if they want
save_file = input('Would you like to save the file?y/n: \n', 's');
if(strncmp(save_file,'y',1))
    uisave();
end
