% This function tries to find the right value which makes the 
% background color the closest to the one in the dataset images
% so that our "predictOneVsAll" function can predict our handwriting correctly.
% If predicted correctly with the number by which the pixel values are incremented,
% the number returned. -1 otherwise.

% img: 20x20 grayscale image cast to double().
% all_theta: theta values for classifiers from the exercise
% y: the actual value that is to be predicted in the image, in this case it is 5.

function p = findRightBgColor(img, all_theta, y)
  % Get max value from the image matrix
  max_value = max(img(:)); 
  % Get min value from the image matrix
  min_value = min(img(:));
  % Set p -1 in case we can't predict the digit successfully so the function returns -1
  p = -1;
  % Map pixel values between 0 and 1
  mapped_img = (img - min_value) / (max_value - min_value) * (-1) + 1;
  % Increment the mapped image matrix 
  % by adding all the values from -0.9 to 1 with a step size of 0.01
  for i = -0.9:0.01:1
    % Increment by i
    temp = mapped_img + i; 
    % Try to predict the new image matrix with incremented pixel values
    prediction = predictOneVsAll(all_theta, temp(:)');
    % If predicted successfully
    if prediction == y
      % Set p to the number that has been added to all pixel values
      p = i;
      break;
    end
  end
end
