% Example usage
x = linspace(-2, 2, 400); % Create a linearly spaced vector from -2 to 2
lower_threshold = -0.0001; % Define the lower threshold
upper_threshold = 0.0001; % Define the upper threshold
y = TernaryActivation(x, lower_threshold, upper_threshold); % Apply the ternary activation function

% Plotting the function
figure;
plot(x, y, 'LineWidth', 2);
title('Ternary Activation Function Plot');
xlabel('Input value (x)');
ylabel('Activated value (y)');
grid on;
axis tight;
ylim([-1.5 1.5]); % Adjust y-axis limits for better visibility
