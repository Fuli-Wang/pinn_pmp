clear;
%% Parameters
% Stewart platform side-centered geometry
base_side_radius     = 225;    % Distance from center to each flat side of the base [mm]
platform_side_radius = 175;    % Distance from center to each flat side of the platform [mm]
z_base    = 0;                 % Z-coordinate of the base plane
z_platform= 456;               % Z-coordinate of the platform plane
num_legs  = 6;                 % Total number of actuators/attachment points
pair_offset_deg = 15;          % ±7.5° per pair

% Base side-center angles: 0°, 120°, 240°
base_side_centers = deg2rad([0, 120, 240]);
% Platform side-center angles: 30°, 150°, 270°
platform_side_centers = base_side_centers + deg2rad(60);

pair_offset_rad = deg2rad(pair_offset_deg);

%% 1) Compute angles for each leg on the base and platform
theta_base = [];
for center_angle = base_side_centers
    theta_base = [theta_base, ...
                  center_angle - pair_offset_rad/2, ...
                  center_angle + pair_offset_rad/2];
end
theta_base = sort(mod(theta_base, 2*pi));

theta_platform = [];
for center_angle = platform_side_centers
    theta_platform = [theta_platform, ...
                      center_angle - pair_offset_rad/2, ...
                      center_angle + pair_offset_rad/2];
end
theta_platform = sort(mod(theta_platform, 2*pi));

%% 2) Compute base and platform attachment points in 3D
base_points = [
    base_side_radius * cos(theta_base)', ...
    base_side_radius * sin(theta_base)', ...
    z_base * ones(num_legs, 1)
];
platform_points = [
    platform_side_radius * cos(theta_platform)', ...
    platform_side_radius * sin(theta_platform)', ...
    z_platform * ones(num_legs, 1)
];

%% 3) Actuator length limits (example)
min_length = 357;  % [mm]
max_length = 571;  % [mm]
% Pose limits
pose_limits = struct( ...
    'x', [-60, 60], ...
    'y', [-60, 60], ...
    'z', [406, 506], ...
    'roll', [-0.2618, 0.2618], ...
    'pitch', [-0.2618, 0.2618], ...
    'yaw', [-0.2618, 0.2618] ...
);

% Generate random samples for dataset
num_samples = 500000;
platform_poses = zeros(num_samples, 6);  % x, y, z, roll, pitch, yaw
actuator_lengths = zeros(num_samples, num_legs);

for i = 1:num_samples
    valid_pose = false;
    while ~valid_pose
        % Generate random pose within limits
        pose_candidate = [
            (pose_limits.x(2) - pose_limits.x(1)) * rand() + pose_limits.x(1), ...
            (pose_limits.y(2) - pose_limits.y(1)) * rand() + pose_limits.y(1), ...
            (pose_limits.z(2) - pose_limits.z(1)) * rand() + pose_limits.z(1), ...
            (pose_limits.roll(2) - pose_limits.roll(1)) * rand() + pose_limits.roll(1), ...
            (pose_limits.pitch(2) - pose_limits.pitch(1)) * rand() + pose_limits.pitch(1), ...
            (pose_limits.yaw(2) - pose_limits.yaw(1)) * rand() + pose_limits.yaw(1)
        ];

        % Compute rotation matrix
        R = eul2rotm([pose_candidate(6), pose_candidate(5), pose_candidate(4)], 'ZYX');

        % Transform platform points to global frame
        platform_center = mean(platform_points, 1); %updated!!!
        platform_points_centered = platform_points - platform_center;
        platform_points_global = (R * platform_points_centered')' + pose_candidate(1:3);

        % Compute actuator lengths
        lengths_candidate = zeros(1, num_legs);
        for j = 1:num_legs
            leg_vector = platform_points_global(j, :) - base_points(j, :);
            lengths_candidate(j) = norm(leg_vector);
        end

        % Validate actuator lengths
        if all(lengths_candidate >= min_length & lengths_candidate <= max_length)
            platform_poses(i, :) = round(pose_candidate, 2);
            actuator_lengths(i, :) = round(lengths_candidate, 2);
            valid_pose = true;
        end
    end
end

% Convert yaw-pitch-roll from radians to degrees for saving
platform_poses(:, 4:6) = rad2deg(platform_poses(:, 4:6));

% Save dataset to text files
save('stewart_data_mm.mat', 'platform_poses', 'actuator_lengths');

platform_poses_file = fopen('platform_poses.txt', 'w');
for i = 1:size(platform_poses, 1)
    fprintf(platform_poses_file, '% 4.4f % 4.4f % 4.4f % 4.4f % 4.4f % 4.4f\n', platform_poses(i, :));
end
fclose(platform_poses_file);

actuator_lengths_file = fopen('actuator_lengths.txt', 'w');
for i = 1:size(actuator_lengths, 1)
    fprintf(actuator_lengths_file, '% 4.4f % 4.4f % 4.4f % 4.4f % 4.4f % 4.4f\n', actuator_lengths(i, :));
end
fclose(actuator_lengths_file);

disp('Data successfully saved.');
