function y = TernaryActivation(x, lower_threshold, upper_threshold)
    % TernaryActivation Function that maps inputs to -1, 0, or 1 based on thresholds.
    % Inputs:
    %   x - Input array
    %   lower_threshold - Lower bound threshold
    %   upper_threshold - Upper bound threshold
    % Outputs:
    %   y - Output array after applying ternary activation
    y = 1.5*tanh(x)+0.3*tanh(-5*x);
%     y = zeros(size(x)); % Initialize output array with zeros
    y(x <= upper_threshold & x>=lower_threshold) = 0; % Set elements to 1 where x is greater than upper_threshold
    % Set elements to -1 where x is less than lower_threshold
    % Elements where x is between the thresholds remain zero
end
