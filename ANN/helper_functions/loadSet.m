function [training_set, test_set, data_set, test_set_length] = loadSet(data_set, split_ratio)
%   This function takes in a dataset and split ratio. It returns a training
%   set and a test set.
% 
%   The ratio entered will be a number between less than or equal to 1 and 
%   greater than 0.

%if there is an incorrect input, go ahead and shut it down
if (split_ratio > 1 || split_ratio <= 0)
    e = MException('MyComponent:noSuchVariable', ...
        'Split ratio %f must be less than 1 and greater than 0\n',split_ratio);
    throw(e)
end
if (size(data_set) == 0)
    e = MException('MyComponent:noSuchVariable', ...
        'There must be data in the set\n');
    throw(e)
end

training_set_length = round(length(data_set) * split_ratio); %training set will be split ratio of the data set
test_set_length = round(length(data_set) * (1 - split_ratio)) - 1; %test set will be the rest of the data set

if ((training_set_length + test_set_length) < length(data_set))
    test_set_length = test_set_length + 1;
end

no_data_cols = size(data_set, 2); %get number of columns in the dataset
training_set = zeros(training_set_length, no_data_cols);
test_set = zeros(test_set_length, no_data_cols); %preallocating for speed--thank you, MATLAB


%https://stackoverflow.com/questions/5444248/random-order-of-rows-matlab#5444297
%code I found for swapping rows... This allows for randomization in data
%
data_set = data_set(randperm(size(data_set,1)),:); %randomize rows in the dataset

index = 1; %used for indexing the dataset across two loops
for i = 1:training_set_length
    training_set(i,1:no_data_cols) = data_set(index,1:no_data_cols);          %sample the data for training set
    index = index + 1;
end

for i = 1: test_set_length
    test_set(i,1:no_data_cols) = data_set(index,1:no_data_cols);              %sample the data for test set
    index = index + 1;
end
