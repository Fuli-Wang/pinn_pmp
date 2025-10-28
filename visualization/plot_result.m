clear
% Load the data
data = load('results_h.txt');

% Separate the joint angles and end positions
translation = data(:, 3);
rotation = data(:, 4:6);
leg_lengths = data(:, 7:12);

% Plot the XYZ
 figure;
 plot(translation, 'LineWidth', 2); % Thicker lines
 title('Platform Height Over Time');
 xlabel('Time (Iterations)');
 ylabel('Platform Height (mm)');
 xlim([0 1000]);  % Adjust this range based on data
 % legend('Platform Height');
 grid on;

% Plot the Euler angles
 figure;
 plot(rotation, 'LineWidth', 2); % Thicker lines
 title('Poses (Euler) Over Time');
 xlabel('Time (Iterations)');
 ylabel('Euler Angles (degrees)');
 xlim([0 1000]);  % Adjust this range based on data
 legend('Roll', 'Pitch', 'Yaw');
 grid on;

% Plot the leg lengths
 figure;
 plot(leg_lengths, 'LineWidth', 2); % Thicker lines
 title('Leg Lengths Over Time');
 xlabel('Time (Iterations)');
 ylabel('Leg Lengths (mm)');
 xlim([0 1000]);  % Adjust this range based on data
 legend('Leg 1', 'Leg 2', 'Leg 3', 'Leg 4', 'Leg 5', 'Leg 6');
 grid on;

% Label the Start Point
% start_point = end_positions(1, :);
% text(start_point(1), start_point(2), start_point(3), 'Start', 'FontSize', 12, 'Color', 'green', 'FontWeight', 'bold');
% 
% % Label the End Point
% end_point = end_positions(end, :);
% text(end_point(1), end_point(2), end_point(3), 'End', 'FontSize', 12, 'Color', 'red', 'FontWeight', 'bold');
