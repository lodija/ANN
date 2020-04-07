function [true_positive, confusion_matrix] = performanceFunction(y, y_hat, confusion_matrix)
%PERFORMANCEFUNCTION takes in the actual value (y) and the expected value
%(y_hat), and returns the total number of true positives (guesses that were
%right. This function also 
% 

true_positive = 0;
    
[~, guessed_indx] = max(y_hat); %get index of highest value of the actual output
[~, correct_output] = max(y);
match = (y(guessed_indx) > 0);  %if the corresponding index is the theoretical output
if(match)
    true_positive = true_positive + 1;  %add 1 for correct guess
end

confusion_matrix(guessed_indx, correct_output) = confusion_matrix(guessed_indx, correct_output) + 1;


%code for finding true negatives, etc; however, for this application, there
%is only one classification (right/wrong)
% for i = 1:length(y)
%     
%     
%     if(y_hat(i) < 0)    % expected value is negative
%         if(y(i) <=  y_hat(i)) % within threshold (correct)
%                 true_negative = true_negative + 1;
%         elseif(y(i) > y_hat(i)) % outside of threshold (incorrect)
%                 false_negative = false_negative + 1; 
%         end
%     elseif(y_hat(i) > 0)    % expected value is positive
% %         if(y(i) >= y_hat(i)) % within threshold (correct)
% %             true_positive = true_positive + 1; 
%         if(y(i) < y_hat(i)) % outside of threshold (incorrect)
%             false_positive = false_positive + 1;
%         end
%     end
% end

