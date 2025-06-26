import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch.autograd

# Kompliance
KFORCE = 50
ITERATION=1000
RAMP_KONSTANT=0.005
RAMP_KONSTANT_Euler=0.001
t_dur=5
J2H=1;

inputL=6
outputL=6

base_radius = 225
platform_radius = 175
hexagon_skew = 0.0
z_base = 0.0
z_platform = 456
shape = 'hexagonal'

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

# Generate points
def generate_base_and_platform_points(platform_shape, base_radius, platform_radius, hexagon_skew, z_base, z_platform, num_legs=6):
    """
    Generate base and platform points arranged in a circle, with platform points skewed.
    """
    if platform_shape == 'circle':
        print("circle")
        theta_base = torch.linspace(0, 2 * torch.pi, num_legs + 1)[:-1]
        theta_platform = theta_base + hexagon_skew #uniform distribution
        #theta_skewed_deg = torch.tensor([15, 45, 135, 165, 255, 285], dtype=torch.float32)
        #theta_platform = theta_skewed_deg * torch.pi / 180.0  # convert to radians

    else: # Generate base and platform points for a Stewart Platform with hexagonal skew applied to platform points.
        print("hexagonal")
        # Compute base pair centers: for num_legs=6, these will be 0, 2π/3, 4π/3.
        base_pair_centers = torch.linspace(0, 2 * torch.pi, num_legs // 2 + 1)[:-1]
        # Platform pair centers are the base pair centers shifted by hexagon_skew.
        platform_pair_centers = base_pair_centers + torch.pi/3

        pair_offset_deg = 15

        # Compute angles for each leg in pairs
        pair_offset_rad = torch.deg2rad(torch.tensor(pair_offset_deg))
        theta_base = []
        theta_platform = []
        for base_center, plat_center in zip(base_pair_centers, platform_pair_centers):
            theta_base.extend([base_center - pair_offset_rad / 2, base_center + pair_offset_rad / 2])
            theta_platform.extend([plat_center - pair_offset_rad / 2, plat_center + pair_offset_rad / 2])

        theta_base = torch.tensor(theta_base, dtype=torch.float32)
        theta_platform = torch.tensor(theta_platform, dtype=torch.float32)

        theta_base = torch.remainder(theta_base, 2 * torch.pi)
        theta_platform = torch.remainder(theta_platform, 2 * torch.pi)
        theta_base, _ = torch.sort(theta_base)
        theta_platform, _ = torch.sort(theta_platform)

    # Generate base points
    base_points = torch.stack([
        base_radius * torch.cos(theta_base),
        base_radius * torch.sin(theta_base),
        z_base * torch.ones(num_legs)
    ], dim=1)

    platform_points = torch.stack([
        platform_radius * torch.cos(theta_platform),
        platform_radius * torch.sin(theta_platform),
        z_platform * torch.ones(num_legs)
    ], dim=1)

    return base_points, platform_points

base_points, platform_points = generate_base_and_platform_points(
    shape, base_radius, platform_radius, hexagon_skew, z_base, z_platform
)
platform_points[:, 2] = 0.0  # platform local frame: z = 0
platform_points_centered = platform_points - platform_points.mean(dim=0, keepdim=True)  # center x/y

#claim model
num_neurons = 256

# Define the pinn model in PyTorch
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

        return lengths

    def get_jacobian(self, inputs):
        inputs = inputs.requires_grad_()
        jac = torch.autograd.functional.jacobian(self.forward, inputs, vectorize=False)
        batch_size = inputs.size(0)
        diag_jac = jac[torch.arange(batch_size), :, torch.arange(batch_size)]
        return diag_jac

#put the Initial position of the platform
def InitializeJan():
    Pose = np.zeros(6)
    Pose[2] = 456

    poseini0 = 0
    poseini1 = 0
    poseini2 = 456
    poseini3 = 0
    poseini4 = 0
    poseini5= 0

    l1_iniIC= 483.2198
    l2_iniIC= 483.2198
    l3_iniIC= 483.2198
    l4_iniIC= 483.2198
    l5_iniIC= 483.2198
    l6_iniIC= 483.2198

    l1_ini = 483.2198
    l2_ini = 483.2198
    l3_ini = 483.2198
    l4_ini = 483.2198
    l5_ini = 483.2198
    l6_ini = 483.2198

    return Pose, poseini0, poseini1, poseini2, poseini3, poseini4, poseini5, l1_iniIC, l2_iniIC, l3_iniIC, l4_iniIC, l5_iniIC, l6_iniIC, l1_ini, l2_ini, l3_ini, l4_ini, l5_ini, l6_ini

def inverse_kinematics(q, model):
    # Convert input to a PyTorch tensor
    u = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
    # Set the model to evaluation mode
    model.eval()
    # Forward pass through the model
    with torch.no_grad():
        a = model(u)
    # Convert the result to a NumPy array
    x = a.numpy()
    return x[0]

def forcefield(w,v):
    j = 0
    length = np.zeros(6)
    tar = np.zeros(6)
    res = np.zeros(6)
    for j in range(6):
        length[j] = w[j]
        tar[j] = v[j]
        res[j] = KFORCE * (tar[j] - length[j])
    ptr = res
    return ptr

def pmp(force, Pose, model):
    ff = np.zeros(6)
    Joint_Field = np.zeros(6)
    Jvel = np.zeros(6)

    for i in range(6):
        ff[i] = force[i]

    JacT =[]

    torch_pose = torch.tensor(Pose, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    #model.eval()
    jacobian = model.get_jacobian(torch_pose)

    Jack = np.squeeze(jacobian.numpy())

    JacT = Jack.T

    Joint_Field[0]=(0-Pose[0])*J2H;# 45 25
    Joint_Field[1]=(0-Pose[1])*J2H; # J2H = 1
    Joint_Field[2]=(456-Pose[2])*J2H;
    Joint_Field[3]=(0-Pose[3])*J2H*1; #50  5
    Joint_Field[4]=(0-Pose[4])*J2H*1;#50  75
    Joint_Field[5]=(0-Pose[5])*J2H*1; #was 30   100

    for a in range(inputL):
        jvelo = 0
        for n in range(outputL):
            jvelo = jvelo+(JacT[a][n]*ff[n])
        Jvel[a] = 0.002*(jvelo+ Joint_Field[a])

    foof = Jvel
    return foof


def GammaDisc(_Time):
    t_ramp=(_Time)*RAMP_KONSTANT
    t_init=0.1
    z=(t_ramp-t_init)/t_dur
    t_win=(t_init+t_dur)-t_ramp

    if t_win>0:
        t_window=1
    else:
        t_window=0
    csi=(6*pow(z,5))-(15*pow(z,4))+(10*pow(z,3)) #6z^5-15z^4+10z^3
    csi_dot=(30*pow(z,4))-(60*pow(z,3))+(30*pow(z,2)) #csi_dot=30z^4-60z^3+30z^2
    prod1=(1/(1.0001-(csi*t_window)))
    prod2=(csi_dot*(1/3)*t_window)
    Gamma=prod1*prod2

    return Gamma

def Gamma_IntDisc(Gar, n):
    k = 1
    a = 0
    sum =Gar[0]
    c = 2
    h = 1
    while k <= (n-1):
        fk = Gar[k]
        c = 6-c
        sum = (sum + c*fk)
        k += 1
    sum=RAMP_KONSTANT*sum/3

    return sum

def Gamma_IntDisc_Euler(Gar, n):
    return RAMP_KONSTANT_Euler * np.sum(Gar[:n])

def MotCon(L1, L2, L3, L4, L5, L6, time, Gam, Pose, q1, q2, q3, q4, q5, q6, janini0, janini1, janini2, janini3, janini4, janini5, model):
    ang = Pose
    nFK = inverse_kinematics(ang, model)
    X_pos = np.zeros(6)
    target = np.zeros(6)

    for i in range(6):
        X_pos[i] = nFK[i]
    po = X_pos

    target[0]=L1
    target[1]=L2
    target[2]=L3
    target[3]=L4
    target[4]=L5
    target[5]=L6

    ta = target
    force = forcefield(po,ta)

    ffield = np.zeros(6)
    for i in range(6):
        ffield[i] = force[i]
    topmp= ffield
    Q_Dot=pmp(topmp, Pose, model)

    JoVel = np.zeros(6)
    for i in range(6):
        JoVel[i]=(Q_Dot[i])*Gam

    q1[time]=JoVel[0]
    j1=q1
    joi1=Gamma_IntDisc(j1,time)
    Pose[0]=joi1+janini0

    q2[time]=JoVel[1]
    j2=q2
    joi2=Gamma_IntDisc(j2,time)
    Pose[1]=joi2+janini1

    q3[time]=JoVel[2]
    j3=q3
    joi3=Gamma_IntDisc(j3,time)
    Pose[2]=joi3+janini2

    q4[time]=JoVel[3]
    j4=q4
    joi4=Gamma_IntDisc(j4,time)
    Pose[3]=joi4+janini3

    q5[time]=JoVel[4]
    j5=q5
    joi5=Gamma_IntDisc(j5,time)
    Pose[4]=joi5+janini4

    q6[time]=JoVel[5]
    j6=q6
    joi6=Gamma_IntDisc(j6,time)
    Pose[5]=joi6+janini5 #(joi6+janini5+ np.pi) % (2 * np.pi) - np.pi

    return Pose, X_pos, q1, q2, q3, q4, q5,q6

def VTGS(LT1, LT2, LT3, LT4, LT5, LT6, XO1, YO2, ZO3, ChoiceAct, MentalSim, WristGraspPose, model):

    results = []

    Pose, janini0, janini1, janini2, janini3, janini4, janini5, l1_iniIC, l2_iniIC, l3_iniIC, l4_iniIC, l5_iniIC, l6_iniIC, l1_ini, l2_ini, l3_ini, l4_ini, l5_ini, l6_ini = InitializeJan()

    fin = np.zeros(6)
    n = 6
    retvalue=0

    if ChoiceAct==0:

        fin = [LT1, LT2, LT3, LT4, LT5, LT6]

        replan=0

        l1_fin=fin[0]
        l2_fin=fin[1]
        l3_fin=fin[2]
        l4_fin=fin[3]
        l5_fin=fin[4]
        l6_fin=fin[5]

        print(" Targets")
        print(l1_fin,l2_fin,l3_fin,l4_fin,l5_fin,l6_fin)

        Gam_Arr1 = np.zeros(ITERATION)
        Gam_Arr2 = np.zeros(ITERATION)
        Gam_Arr3 = np.zeros(ITERATION)
        Gam_Arr4 = np.zeros(ITERATION)
        Gam_Arr5 = np.zeros(ITERATION)
        Gam_Arr6 = np.zeros(ITERATION)

        q1 = np.zeros(ITERATION)
        q2 = np.zeros(ITERATION)
        q3 = np.zeros(ITERATION)
        q4 = np.zeros(ITERATION)
        q5 = np.zeros(ITERATION)
        q6 = np.zeros(ITERATION)

        final_length = np.zeros(n)

        for time in range(ITERATION):
            Gam=GammaDisc(time);

            #Target Generation

            inter_l1=(l1_fin-l1_ini)*Gam
            Gam_Arr1[time]=inter_l1
            Gar1=Gam_Arr1
            l1_ini=Gamma_IntDisc(Gar1,time)+l1_iniIC

            inter_l2=(l2_fin-l2_ini)*Gam
            Gam_Arr2[time]=inter_l2
            Gar2=Gam_Arr2
            l2_ini=Gamma_IntDisc(Gar2,time)+l2_iniIC

            inter_l3=(l3_fin-l3_ini)*Gam
            Gam_Arr3[time]=inter_l3
            Gar3=Gam_Arr3
            l3_ini=Gamma_IntDisc(Gar3,time)+l3_iniIC

            inter_l4=(l4_fin-l4_ini)*Gam
            Gam_Arr4[time]=inter_l4
            Gar4=Gam_Arr4
            l4_ini=Gamma_IntDisc(Gar4,time)+l4_iniIC

            inter_l5=(l5_fin-l5_ini)*Gam
            Gam_Arr5[time]=inter_l5
            Gar5=Gam_Arr5
            l5_ini=Gamma_IntDisc(Gar5,time)+l5_iniIC

            inter_l6=(l6_fin-l6_ini)*Gam
            Gam_Arr6[time]=inter_l6
            Gar6=Gam_Arr6
            l6_ini=Gamma_IntDisc(Gar6,time)+l6_iniIC

            Pose, X_pos, q1, q2, q3, q4, q5, q6 = MotCon(l1_ini, l2_ini, l3_ini, l4_ini, l5_ini, l6_ini, time, Gam, Pose, q1, q2, q3, q4, q5, q6, janini0, janini1, janini2, janini3, janini4, janini5, model)

            results.append([Pose[0],Pose[1],Pose[2],Pose[3],Pose[4],Pose[5],l1_ini, l2_ini, l3_ini, l4_ini, l5_ini, l6_ini])

        final_length =[X_pos[0],X_pos[1],X_pos[2],X_pos[3],X_pos[4],X_pos[5]]

        konst=1
        ang1=konst*Pose[0]
        ang2=konst*Pose[1]
        ang3=konst*Pose[2]
        ang4=konst*Pose[3]
        ang5=konst*Pose[4]
        ang6=konst*Pose[5]
        final_length = [float(x) for x in final_length]
        print("\n Platform Poses: ",ang1,ang2,ang3,ang4,ang5,ang6)
        print("\n\n FINAL SOLUTION: ",final_length)
        print("\n\n Error: ",np.linalg.norm(np.array(fin) - np.array(final_length)))
        np.savetxt('results.txt', results, fmt='%f') #The file records the whole motion (Joint angles' change and end-effector's trajectory)
        #time.sleep(1)

def TargGenSMo(model):

    target_length = np.zeros(6)

    try:
        f = open('target_length.txt') #here is your target position file(should refer the workspce)
        matrix = f.read().split()
        target_length[0] = matrix[0]
        target_length[1] = matrix[1]
        target_length[2] = matrix[2]
        target_length[3] = matrix[3]
        target_length[4] = matrix[4]
        target_length[5] = matrix[5]
    except:
        print("Oops!  Cannot find the target file.")

    VTGS(target_length[0], target_length[1], target_length[2], target_length[3], target_length[4], target_length[5], 0,0,0,0,0,0, model)

if __name__ == "__main__":
    model = PMP(num_legs=6)
    model.load_state_dict(torch.load('best_model.pth'))
    #model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))# Load the model on CPU
    TargGenSMo(model)
