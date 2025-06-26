import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import time
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#define your robot configuration
base_radius = 225
platform_radius = 175
hexagon_skew = 0.0
z_base = 0.0
z_platform = 456
shape = 'hexagonal' #we set two platforms, please use the MATLAB visualization code to see the structure

# Generate fixed points based on robot configuration, can change the distribution of base and platform points
def generate_base_and_platform_points(platform_shape, base_radius, platform_radius, hexagon_skew, z_base, z_platform, num_legs=6, device=device):
    """
    Generate base and platform points arranged in a circle, with platform points skewed.
    """
    if platform_shape == 'circle':
        print("circle")
        theta_base = torch.linspace(0, 2 * torch.pi, num_legs + 1, device=device)[:-1]
        theta_platform = theta_base + hexagon_skew #uniform distribution
        #theta_skewed_deg = torch.tensor([15, 45, 135, 165, 255, 285], dtype=torch.float32, device=device)
        #theta_platform = theta_skewed_deg * torch.pi / 180.0  # convert to radians


    else: # Generate base and platform points for a Stewart Platform with hexagonal skew applied to platform points.
        print("hexagonal")
        # Compute base pair centers: for num_legs=6, these will be 0, 2π/3, 4π/3.
        base_pair_centers = torch.linspace(0, 2 * torch.pi, num_legs // 2 + 1, device=device)[:-1]
        # Platform pair centers are the base pair centers shifted by hexagon_skew.
        platform_pair_centers = base_pair_centers + torch.pi/3

        pair_offset_deg = 15

        # Compute angles for each leg in pairs
        pair_offset_rad = torch.deg2rad(torch.tensor(pair_offset_deg, device=device))
        theta_base = []
        theta_platform = []
        for base_center, plat_center in zip(base_pair_centers, platform_pair_centers):
            theta_base.extend([base_center - pair_offset_rad / 2, base_center + pair_offset_rad / 2])
            theta_platform.extend([plat_center - pair_offset_rad / 2, plat_center + pair_offset_rad / 2])

        theta_base = torch.tensor(theta_base, dtype=torch.float32, device=device)
        theta_platform = torch.tensor(theta_platform, dtype=torch.float32, device=device)

        theta_base = torch.remainder(theta_base, 2 * torch.pi)
        theta_platform = torch.remainder(theta_platform, 2 * torch.pi)
        theta_base, _ = torch.sort(theta_base)
        theta_platform, _ = torch.sort(theta_platform)

    # Generate base points
    base_points = torch.stack([
        base_radius * torch.cos(theta_base),
        base_radius * torch.sin(theta_base),
        z_base * torch.ones(num_legs, device=device)
    ], dim=1)

    platform_points = torch.stack([
        platform_radius * torch.cos(theta_platform),
        platform_radius * torch.sin(theta_platform),
        z_platform * torch.ones(num_legs, device=device)
    ], dim=1)

    return base_points, platform_points

base_points, platform_points = generate_base_and_platform_points(
    shape, base_radius, platform_radius, hexagon_skew, z_base, z_platform
)
platform_points[:, 2] = 0.0  # platform local frame: z = 0
platform_points_centered = platform_points - platform_points.mean(dim=0, keepdim=True)  # center x/y

def compute_batch_rotation_matrix(euler_angles):
    """
    Compute batch rotation matrices from Euler angles (ZYX convention).
    Input:  [batch_size, 3] -> roll (x), pitch (y), yaw (z)
    Output: [batch_size, 3, 3] rotation matrices
    """
    roll  = euler_angles[:, 0]  # rotation around x-axis
    pitch = euler_angles[:, 1]  # rotation around y-axis
    yaw   = euler_angles[:, 2]  # rotation around z-axis

    # Compute cosines and sines
    cos_r = torch.cos(roll)
    sin_r = torch.sin(roll)
    cos_p = torch.cos(pitch)
    sin_p = torch.sin(pitch)
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)

    # Rotation matrices components
    batch_size = euler_angles.shape[0]

    R = torch.zeros((batch_size, 3, 3), dtype=euler_angles.dtype, device=euler_angles.device)

    R[:, 0, 0] = cos_y * cos_p
    R[:, 0, 1] = cos_y * sin_p * sin_r - sin_y * cos_r
    R[:, 0, 2] = cos_y * sin_p * cos_r + sin_y * sin_r

    R[:, 1, 0] = sin_y * cos_p
    R[:, 1, 1] = sin_y * sin_p * sin_r + cos_y * cos_r
    R[:, 1, 2] = sin_y * sin_p * cos_r - cos_y * sin_r

    R[:, 2, 0] = -sin_p
    R[:, 2, 1] = cos_p * sin_r
    R[:, 2, 2] = cos_p * cos_r

    return R

# Define the PINN model
class PMP(nn.Module):
    def __init__(self, num_legs=6, platform_points=platform_points_centered, hidden_dim=256):
        super(PMP, self).__init__()
        self.num_legs = num_legs
        self.register_buffer('platform_points_local', platform_points)

        self.fc_output = nn.Sequential(
            nn.Linear(num_legs * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),  # New hidden layer
            nn.SiLU(),
            nn.Linear(hidden_dim, num_legs),
            nn.Softplus()             # Final ReLU or Softplus activation to ensure non-negative outputs
        )

    def forward(self, pose_candidate):

        translation = pose_candidate[:, :3]
        euler_angles_deg = pose_candidate[:, [3, 4, 5]]      # roll, pitch, yaw (swap order to ZYX)
        euler_angles = torch.deg2rad(euler_angles_deg)  # convert to radians
        R = compute_batch_rotation_matrix(euler_angles)

        batch_size = pose_candidate.shape[0]

        platform_points_rotated = torch.bmm(R, self.platform_points_local.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2)).transpose(1, 2)

        platform_points_global = platform_points_rotated + translation.unsqueeze(1)

        platform_points_flat = platform_points_global.reshape(platform_points_global.size(0), -1)
        lengths = self.fc_output(platform_points_flat)

        return platform_points_global, lengths

    def get_jacobian(self, inputs): # can be used for jacobian verification if necessary
        inputs = inputs.requires_grad_()
        outputs = self.forward(inputs)
        jacobian = []
        for i in range(outputs.shape[1]):
            grad_outputs = torch.zeros_like(outputs)
            grad_outputs[:, i] = 1.0
            jacobian.append(torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs, retain_graph=True)[0])
        return torch.stack(jacobian, dim=1)

# Load data function
def load_data(file_path, num_rows, num_cols):
    data = np.zeros((num_rows, num_cols), dtype=np.float32)
    try:
        with open(file_path, 'r') as f:
            matrix = f.read().split()
            for i in range(num_rows):
                for j in range(num_cols):
                    data[i, j] = float(matrix[i * num_cols + j])
    except FileNotFoundError:
        print(f"Oops! Cannot find the file {file_path}.")
    return data


# Initialize model, loss function, and optimizer
model = PMP(num_legs=6, platform_points=platform_points_centered).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

def custom_loss(outputs,platform_points_global, base_points, epoch):
    # Physics constraint: Enforce length consistency
    base_points = base_points.to(platform_points_global.device)
    leg_vectors = platform_points_global - base_points.unsqueeze(0)
    predicted_lengths = torch.norm(leg_vectors, dim=2)

    physics_loss = F.mse_loss(outputs, predicted_lengths)

    return  physics_loss

# Load data
num_data = 500000
x = load_data('platform_poses.txt', num_data, 6)

# Split and convert data
x_train, x_val = train_test_split(x, test_size=0.2, random_state=42)
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
x_val = torch.tensor(x_val, dtype=torch.float32).to(device)

# Training parameters
batch_size = 256
epochs = 200
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
losses, val_losses = [], []
num_batches = len(x_train) // batch_size
best_val_loss, patience, wait = float('inf'), 100, 0

# Training loop
start_time = time.time()
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    permutation = torch.randperm(x_train.size(0))
    for i in range(0, x_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x = x_train[indices]

        platform_points_global, outputs = model(batch_x)
        loss = custom_loss(outputs, platform_points_global, base_points, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        platform_points_global_val, outputs_val = model(x_val)
        val_loss = custom_loss(outputs_val, platform_points_global_val, base_points, epoch).item()

    scheduler.step()
    losses.append(epoch_loss / num_batches)
    val_losses.append(val_loss)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: loss = {epoch_loss/num_batches:.6f}, val_loss = {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

# Save model and results
end_time = time.time()
print(f"Total training time: {(end_time - start_time) / 3600:.2f} hours")
torch.save(model.state_dict(), 'best_model.pth')

with open('losses.txt', 'w') as f:
    for epoch, (loss, val_loss) in enumerate(zip(losses, val_losses), start=1):
        f.write(f"Epoch {epoch}: Loss: {loss:.6f}, Val Loss: {val_loss:.6f}\n")
