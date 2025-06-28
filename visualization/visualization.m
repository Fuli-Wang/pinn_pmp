clear; clc;

%% USER PARAMETERS
base_side_radius     = 225;  % distance from center to each flat side of the base
platform_side_radius = 175;  % distance from center to each flat side of the platform
z_base      = 0;
z_platform  = 456;
num_legs    = 6;
pair_offset_deg = 15;       % ±7.5° per pair

% Input the pose you want to see
pose_input = [-0.16367666776497108 -0.6596727518850166 423.72335568362746 -7.069073827644413 7.809277820024048 -7.584001227298295]; % [X Y Z roll pitch yaw] millimeter and degree
pose_input(:, 4:6) = deg2rad(pose_input(:, 4:6)); % transfer degree to radian
R = eul2rotm([pose_input(6), pose_input(5), pose_input(4)], 'ZYX');
z_platform = pose_input(3);
T = pose_input(1:3);

% Base side-center angles: 0°, 120°, 240°
base_pair_centers = deg2rad([0, 120, 240]);
% Platform side-center angles: 30°, 150°, 270°
platform_pair_centers = base_pair_centers + deg2rad(60);

% Convert to radians
pair_offset_rad = deg2rad(pair_offset_deg);

%% 1) ACTUATOR ANGLES
%  -- base actuators
theta_base = [];
for center_angle = base_pair_centers
    theta_base = [theta_base, ...
                  center_angle - pair_offset_rad/2, ...
                  center_angle + pair_offset_rad/2];
end
theta_base = mod(theta_base, 2*pi);
theta_base = sort(theta_base);

%  -- platform actuators
theta_top = [];
for center_angle = platform_pair_centers
    theta_top = [theta_top, ...
                 center_angle - pair_offset_rad/2, ...
                 center_angle + pair_offset_rad/2];
end
theta_top = mod(theta_top, 2*pi);
theta_top = sort(theta_top);

%% 2) ACTUATOR POINTS in 3D
%  Here, base_side_radius & platform_side_radius
%  are used to place the actuators at the side centers.
base_points_3D = [
    base_side_radius * cos(theta_base)', ...
    base_side_radius * sin(theta_base)', ...
    z_base * ones(num_legs,1)
];

platform_points_3D = [
    platform_side_radius * cos(theta_top)', ...
    platform_side_radius * sin(theta_top)', ...
    z_platform * ones(num_legs,1)
];



%% 3) HEX CORNER RADIUS
%  For a regular hex, r_corner = r_side / cos(30°).
corner_scale = 1 / cosd(30);  % ~1.1547
base_corner_radius = base_side_radius * corner_scale;
platform_corner_radius = platform_side_radius * corner_scale;

%% 4) DRAW BASE HEX CORNERS
%  If the side centers are at 0°, 60°, 120°, etc.,
%  the corners are at 30°, 90°, 150°, etc.
theta_hex_base_deg = [30, 90, 150, 210, 270, 330, 30];
theta_hex_base_rad = deg2rad(theta_hex_base_deg);
base_hex_3D = [
    base_corner_radius * cos(theta_hex_base_rad)', ...
    base_corner_radius * sin(theta_hex_base_rad)', ...
    z_base * ones(numel(theta_hex_base_rad),1)
];

%% 5) DRAW PLATFORM HEX CORNERS
%  Similarly, if the platform side centers are at 30°, 90°, etc.,
%  then its corners are at 60°, 120°, 180°, etc.
theta_hex_top_deg = [30, 90, 150, 210, 270, 330, 30];
theta_hex_top_rad = deg2rad(theta_hex_top_deg);
platform_hex_3D = [
    platform_corner_radius * cos(theta_hex_top_rad)', ...
    platform_corner_radius * sin(theta_hex_top_rad)', ...
    z_platform * ones(numel(theta_hex_top_rad),1)
];

all_platform_pts = [platform_points_3D; platform_hex_3D];
platform_center = mean(all_platform_pts, 1);
platform_points_centered = platform_points_3D - platform_center;
platform_hex_centered    = platform_hex_3D    - platform_center;

platform_points_transformed = (R * platform_points_centered')' + T;
platform_hex_transformed    = (R * platform_hex_centered')'    + T;


%% 6) VISUALIZATION
figure('Name','Stewart Platform - Side-Center Radii with Larger Corners');
hold on; grid on; axis equal;

% -- Base hex outline (blue)
plot3(base_hex_3D(:,1), base_hex_3D(:,2), base_hex_3D(:,3), ...
      'b-', 'LineWidth',2, 'DisplayName','Base Hex');

% -- Platform hex outline (red)
plot3(platform_hex_transformed(:,1), platform_hex_transformed(:,2), platform_hex_transformed(:,3), ...
      'r-', 'LineWidth',2, 'DisplayName','Platform Hex');

% -- Base actuator points (blue circles)
plot3(base_points_3D(:,1), base_points_3D(:,2), base_points_3D(:,3), ...
      'bo','MarkerSize',8, 'DisplayName','Base Actuators');

% -- Platform actuator points (red circles)
plot3(platform_points_transformed(:,1), platform_points_transformed(:,2), platform_points_transformed(:,3), ...
      'ro','MarkerSize',8, 'DisplayName','Platform Actuators');

% -- Connect each base actuator to the corresponding platform actuator
for i = 1:num_legs
    plot3([base_points_3D(i,1), platform_points_transformed(i,1)], ...
          [base_points_3D(i,2), platform_points_transformed(i,2)], ...
          [base_points_3D(i,3), platform_points_transformed(i,3)], ...
           'LineWidth',1);
end

xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
title('Stewart Platform: Hex Corners Outward, Actuators on Side Centers');
% legend('Location','best');
view(3);
hold off;
