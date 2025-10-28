#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel robot controller — Robot #3 (Hexagonal skew type)
# This script implements the proposed PINN-PMP framework used in our paper.
# It records all motion and evaluation data (joint/leg trajectories, reference
# values, tracking errors, and control signals) into CSV files for analysis.
# These additional evaluations increase the total runtime compared to a
# standard motion generation loop.
======================================================================
Geometry & initial pose taken from legacy pmp_parallel.py:
  base_radius=225, platform_radius=175, z_platform=456,
  shape='hexagonal', pair_offset_deg=15°, l_ini=483.2198 mm.

Outputs
-------
- results.txt        -> pose(6)+L(6)
- results_head.csv   -> full log with meta header
"""

import argparse, os, numpy as np, torch, torch.nn as nn

# ===== Robot parameters =====
BASE_RADIUS_DEF, PLATFORM_RADIUS_DEF = 225.0, 175.0
Z_BASE_DEF, Z_PLATFORM_DEF = 0.0, 456.0
POSE0_DEFAULT = [0.0, 0.0, Z_PLATFORM_DEF, 0.0, 0.0, 0.0]
INIT_LEG_LEN = 483.2198

ITERATION_DEFAULT, DT_DEFAULT, SUBMV_T_DEFAULT = 1000, 0.004, 1.2
KP_DEF_SCALAR, BQ_DIAG_DEFAULT, LAM2_DEFAULT = 100.0, [0.08,0.08,0.08,0.06,0.06,0.06], 1e-4
KQ_DEFAULT = [0,0,0,0.0,0.0,0.0]
TRAJ_DEF = "minjerk"

# ===== Helper functions =====
def _model_device(model):
    try: return next(model.parameters()).device
    except StopIteration: return torch.device("cpu")

def min_jerk_s(t,T): tau=np.clip(t/max(T,1e-9),0,1); return 10*tau**3-15*tau**4+6*tau**5
def dls_pinv(J,lam2=1e-4): JT=J.T; I=np.eye(J.shape[0]); return JT@np.linalg.inv(J@JT+lam2*I)

# ===== Geometry & model =====
def compute_batch_rotation_matrix(euler):
    r,p,y=euler[:,0],euler[:,1],euler[:,2]
    cr,sr,cp,sp,cy,sy=torch.cos(r),torch.sin(r),torch.cos(p),torch.sin(p),torch.cos(y),torch.sin(y)
    B=euler.shape[0]; R=torch.zeros((B,3,3),dtype=euler.dtype,device=euler.device)
    R[:,0,0]=cy*cp; R[:,0,1]=cy*sp*sr-sy*cr; R[:,0,2]=cy*sp*cr+sy*sr
    R[:,1,0]=sy*cp; R[:,1,1]=sy*sp*sr+cy*cr; R[:,1,2]=sy*sp*cr-cy*sr
    R[:,2,0]=-sp;  R[:,2,1]=cp*sr;  R[:,2,2]=cp*cr; return R

def generate_base_and_platform_points(
    base_radius=BASE_RADIUS_DEF,
    platform_radius=PLATFORM_RADIUS_DEF,
    z_base=Z_BASE_DEF,
    z_platform=Z_PLATFORM_DEF,
    device=None,
):
    """Generate base/platform anchor points for hexagonal-pair layout with ±15° offsets."""
    dtype = torch.float32
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_legs = 6
    pair_offset = torch.deg2rad(torch.tensor(15.0, dtype=dtype, device=device))   # ±15°
    centers_base = torch.arange(3, dtype=dtype, device=device) * (2.0 * torch.pi / 3.0)
    centers_plat = centers_base + (torch.pi / 3.0)

    theta_b = torch.empty(num_legs, dtype=dtype, device=device)
    theta_p = torch.empty(num_legs, dtype=dtype, device=device)

    idx = 0
    for cb, cp in zip(centers_base, centers_plat):
        theta_b[idx]   = cb - pair_offset / 2
        theta_b[idx+1] = cb + pair_offset / 2
        theta_p[idx]   = cp - pair_offset / 2
        theta_p[idx+1] = cp + pair_offset / 2
        idx += 2

    theta_b = torch.remainder(theta_b, 2.0 * torch.pi)
    theta_p = torch.remainder(theta_p, 2.0 * torch.pi)
    theta_b, _ = torch.sort(theta_b)
    theta_p, _ = torch.sort(theta_p)

    base = torch.stack([
        torch.tensor(base_radius,    dtype=dtype, device=device) * torch.cos(theta_b),
        torch.tensor(base_radius,    dtype=dtype, device=device) * torch.sin(theta_b),
        torch.tensor(z_base,         dtype=dtype, device=device).expand(num_legs),
    ], dim=1)

    plat = torch.stack([
        torch.tensor(platform_radius, dtype=dtype, device=device) * torch.cos(theta_p),
        torch.tensor(platform_radius, dtype=dtype, device=device) * torch.sin(theta_p),
        torch.tensor(z_platform,      dtype=dtype, device=device).expand(num_legs),
    ], dim=1)

    plat[:, 2] = 0.0
    return base, plat


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_points, platform_points = generate_base_and_platform_points(device=device)
platform_points_centered = platform_points - platform_points.mean(dim=0, keepdim=True)


class PMP(nn.Module):
    def __init__(self,num_legs=6,hidden_dim=256):
        super().__init__()
        self.register_buffer("platform_points_local",platform_points_centered)
        self.fc_output=nn.Sequential(nn.Linear(num_legs*3,hidden_dim),nn.SiLU(),
                                     nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),
                                     nn.Linear(hidden_dim,num_legs),nn.Softplus())
    def forward(self, pose_candidate):
        #dtype / device
        dtype = pose_candidate.dtype
        device = pose_candidate.device

        translation = pose_candidate[:, :3]
        euler_deg   = pose_candidate[:, 3:6]
        euler_rad   = torch.deg2rad(euler_deg).to(dtype=dtype, device=device)

        R = compute_batch_rotation_matrix(euler_rad)     # [B,3,3]
        B = pose_candidate.shape[0]

        pts_local = self.platform_points_local.to(dtype=dtype, device=device)
        pts_local = pts_local.unsqueeze(0).expand(B, -1, -1)         # [B,6,3]

        pts_rot = torch.bmm(R, pts_local.transpose(1, 2)).transpose(1, 2)  # [B,6,3]
        pts_global = pts_rot + translation.unsqueeze(1)
        flat = pts_global.reshape(B, -1)
        lengths = self.fc_output(flat)
        return lengths

    def get_jacobian(self,x):
        x=x.detach().clone().requires_grad_(True); y=self.forward(x)
        rows=[torch.autograd.grad(y[0,i],x,retain_graph=True)[0][0] for i in range(6)]
        return torch.stack(rows,0)

# ===== Initialization =====
def InitializeJan():
    Pose=np.array(POSE0_DEFAULT); L_iniIC=np.full(6,INIT_LEG_LEN); L_ini=L_iniIC.copy()
    return Pose,*Pose.tolist(),*L_iniIC.tolist(),*L_ini.tolist()

def geom_lengths_from_pose(pose_tensor):
    dtype = pose_tensor.dtype
    dev = pose_tensor.device
    t = pose_tensor[:3]  # x,y,z
    euler_deg = pose_tensor[3:6]
    euler_rad = torch.deg2rad(euler_deg).to(dtype=dtype, device=dev)

    R = compute_batch_rotation_matrix(euler_rad.unsqueeze(0))[0]  # [3,3]
    pts_local = platform_points_centered.to(dtype=dtype, device=dev)          # [6,3]
    pts_rot   = (R @ pts_local.T).T                                           # [6,3]
    pts_world = pts_rot + t                                                   # [6,3]

    base = base_points.to(dtype=dtype, device=dev)                            # [6,3]
    diffs = pts_world - base
    lens  = torch.linalg.norm(diffs, dim=1)                                   # [6]
    return lens


def _metrics_setpoint(e_set):
    abs_e = np.abs(e_set)
    rmse  = float(np.sqrt(np.mean(e_set**2)))
    mae   = float(np.mean(abs_e))
    mabs  = float(np.max(abs_e))
    l2    = float(np.linalg.norm(e_set, ord=2))
    return rmse, mae, mabs, l2


def _append_row_to_table(row_dict, fname="setpoint_table.csv"):
    cols = [
        # target
        *[f"tar_L{i+1}" for i in range(6)],
        # IK legs
        *[f"ik_L{i+1}"  for i in range(6)],
        # error
        *[f"e_set{i+1}" for i in range(6)],
        "RMSE","MAE","MaxAbs","L2","condJ",
        # pose
        "x","y","z","roll","pitch","yaw",
        # control/meta
        "mapping","kp_or_vec","lam2","bq_diag","kq",
        "dt","steps","submv_T","traj","model"
    ]

    vals = []
    vals += list(row_dict["tar_L"])
    vals += list(row_dict["ik_L"])
    vals += list(row_dict["e_set"])
    vals += [row_dict["RMSE"], row_dict["MAE"], row_dict["MaxAbs"], row_dict["L2"], row_dict["condJ"]]
    vals += list(row_dict["pose"])
    vals += [
        row_dict["mapping"], row_dict["kp_or_vec"], row_dict["lam2"],
        " ".join(map(str, row_dict["bq_diag"])), " ".join(map(str, row_dict["kq"])),
        row_dict["dt"], row_dict["steps"], row_dict["submv_T"], row_dict["traj"], row_dict["model"]
    ]
    need_header = not os.path.exists(fname)
    with open(fname, "a", encoding="utf-8") as f:
        if need_header:
            f.write(",".join(cols) + "\n")
        f.write(",".join(str(v) for v in vals) + "\n")

# ===== Control core =====
def core_step(pose,L_ref,model,args):
    device=pose.device
    with torch.no_grad():
        L_cur_t = geom_lengths_from_pose(pose)      # torch[6]
        L_cur   = L_cur_t.detach().cpu().numpy()           # np[6]
    Kp=np.diag(args.kp_vec) if args.kp_vec is not None else np.eye(6)*args.kp
    F=Kp@(np.asarray(L_ref)-L_cur)
    J=model.get_jacobian(pose.unsqueeze(0)).cpu().numpy()
    if args.use_jt: qdot=J.T@F; N=np.eye(6)-J.T@np.linalg.pinv(J)
    else: Jp=dls_pinv(J,args.lam2); qdot=Jp@F; N=np.eye(6)-(Jp@J)
    q=np.array(pose.cpu()); q_post=np.array(args.pose_ref or args.pose0)
    Kq=np.diag(args.kq); qdot_post=N@(Kq@(q_post-q))
    BQ=np.diag(args.bq_diag); qdot=(np.eye(6)-BQ)@(qdot+qdot_post)
    return pose+torch.tensor(qdot*args.dt,dtype=pose.dtype,device=device),L_cur,F,qdot

# ===== Controller runner =====
def run_controller(model,args,target_lengths):
    device=_model_device(model)
    pose = torch.tensor(np.array(args.pose0, dtype=np.float32), device=device)
    with torch.no_grad():
        L0_t = geom_lengths_from_pose(pose)      # torch[6]
        L0 = L0_t.detach().cpu().numpy()           # np[6]
    logs=[];
    for i in range(args.steps):
        t=i*args.dt; s=min_jerk_s(t,args.submv_T)
        L_ref=L0+s*(np.array(target_lengths)-L0)
        pose,L_cur,F,qdot=core_step(pose,L_ref,model,args)
        logs.append([t,*L_cur,*L_ref,*F,*qdot,*pose.cpu().numpy()])
    arr=np.asarray(logs)
    np.savetxt("results_h.txt",np.hstack([arr[:,-6:],arr[:,1:7]]),fmt="%f")
    header=_meta(args)+"\n"+_csv_head()
    np.savetxt("results_head.csv",arr,fmt="%f",header=header,comments="")
    err = np.linalg.norm(np.array(target_lengths) - L_cur)
    print("[info] Simulation completed.")
    print(f"[info] Final pose (mm,deg): {pose.cpu().numpy()}")
    print(f"[info] Final leg lengths (NN, mm): {L_cur}")
    print(f"[info] Target leg lengths (mm): {target_lengths}")
    print(f"[info] Final leg-length error (NN vs target) L2 = {err:.6f} mm")

    with torch.no_grad():
        L_ik = geom_lengths_from_pose(pose).cpu().numpy()   # [6]

    e_set = L_ik - np.array(target_lengths)                 # 6-dim setpoint error
    rmse, mae, mabs, l2 = _metrics_setpoint(e_set)

    J = model.get_jacobian(pose.unsqueeze(0)).cpu().numpy()  # [6,6]
    try:
        condJ = float(np.linalg.cond(J))
    except np.linalg.LinAlgError:
        condJ = float("inf")

    e_str = " ".join(f"{v:+.4f}" for v in e_set)
    print("[info] --- Setpoint IK check (geometry) ---")
    print(f"[info] IK leg lengths (mm): {np.array2string(L_ik, precision=4, separator=', ')}")
    print(f"[info] e_set = IK - target (mm): [{e_str}]")
    print(f"[info] Metrics: RMSE={rmse:.4f}  MAE={mae:.4f}  MaxAbs={mabs:.4f}  L2={l2:.4f}  cond(J)={condJ:.2e}")

    # write setpoint_table.csv
    mapping = "JT" if args.use_jt else "DLS"
    kp_or_vec = ("vec" if args.kp_vec is not None else f"{args.kp}")
    row = {
        "tar_L": np.array(target_lengths, dtype=float),
        "ik_L":  L_ik.astype(float),
        "e_set": e_set.astype(float),
        "RMSE": rmse, "MAE": mae, "MaxAbs": mabs, "L2": l2, "condJ": condJ,
        "pose": pose.cpu().numpy().astype(float),
        "mapping": mapping, "kp_or_vec": kp_or_vec, "lam2": float(args.lam2),
        "bq_diag": list(args.bq_diag), "kq": list(args.kq),
        "dt": float(args.dt), "steps": int(args.steps), "submv_T": float(args.submv_T),
        "traj": args.traj, "model": args.model
    }
    _append_row_to_table(row, fname="setpoint_table.csv")

    print("[info] Files saved: results_head.csv, setpoint_table.csv")
    return pose.cpu().numpy(), L_cur

def _csv_head():
    return ("time,"+",".join([f"L{i+1}" for i in range(6)])+","+
            ",".join([f"Lref{i+1}" for i in range(6)])+","+
            ",".join([f"F{i+1}" for i in range(6)])+","+
            ",".join([f"qdot{i+1}" for i in range(6)])+",x,y,z,roll,pitch,yaw")
def _meta(a):
    return (f"# meta: model={a.model}, steps={a.steps}, dt={a.dt}, traj={a.traj}, "
            f"kp={'vec' if a.kp_vec else a.kp}, use_jt={a.use_jt}, lam2={a.lam2}, "
            f"bq={list(a.bq_diag)}, kq={list(a.kq)}, "
            f"geom(baseR={BASE_RADIUS_DEF},platR={PLATFORM_RADIUS_DEF},zPlat={Z_PLATFORM_DEF}), "
            f"pose0={list(a.pose0)}, pose_ref={list(a.pose_ref) if a.pose_ref else None})")

# ===== CLI =====
def _parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",type=str,default="best_model.pth")
    ap.add_argument("--target",type=float,nargs=6,default=[415.46, 433.71, 465.28, 484.94, 455.51, 463.25])
    ap.add_argument("--pose0",type=float,nargs=6,default=POSE0_DEFAULT)
    ap.add_argument("--pose-ref",dest="pose_ref",type=float,nargs=6)
    ap.add_argument("--steps",type=int,default=ITERATION_DEFAULT)
    ap.add_argument("--dt",type=float,default=DT_DEFAULT)
    ap.add_argument("--submv-T",dest="submv_T",type=float,default=SUBMV_T_DEFAULT)
    ap.add_argument("--traj",choices=["minjerk","vtgs"],default=TRAJ_DEF)
    ap.add_argument("--kp",type=float,default=KP_DEF_SCALAR)
    ap.add_argument("--kp-vec",type=float,nargs=6)
    ap.add_argument("--use-jt",action="store_true")
    ap.add_argument("--lam2",type=float,default=LAM2_DEFAULT)
    ap.add_argument("--bq-diag",type=float,nargs=6,default=BQ_DIAG_DEFAULT)
    ap.add_argument("--kq",type=float,nargs=6,default=KQ_DEFAULT)
    return ap

def main():
    torch.set_default_dtype(torch.float32)
    a=_parser().parse_args()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m=PMP().to(dev); m.load_state_dict(torch.load(a.model,map_location="cpu")); m.eval()
    print(f"[info] Using device: {dev}")
    print(f"[info] Model: {a.model}")
    print(f"[info] Geometry baseR={BASE_RADIUS_DEF}, platR={PLATFORM_RADIUS_DEF}, zPlat={Z_PLATFORM_DEF}")
    print(f"[info] Steps={a.steps} dt={a.dt} traj={a.traj}")
    print(f"[info] Mapping={'J^T' if a.use_jt else 'DLS'} lam2={a.lam2}")
    print(f"[info] Gains kp={a.kp} bq={a.bq_diag} kq={a.kq}")
    print(f"[info] pose0={a.pose0}")
    run_controller(m,a,np.array(a.target))
    print("[info] done.")

if __name__=="__main__": main()


