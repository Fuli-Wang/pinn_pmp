#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMP controller for Delta robot (3-DoF, KR 3 D1200 compatible)
- Loads a trained PINN: input T=[x,y,z] -> output L=[L1,L2,L3]
- Uses min-jerk timing, explicit Euler integration, pose-space damping

  • After the run, compute leg lengths again via analytic FK from final pose
    (using the same KR3-D1200 geometry as in pinntrain_delta.py), and print
    side-by-side with the model-predicted lengths, plus errors and RMSE.
  • Append L_fk1..L_fk3 to results_head.csv for auditing.
  • These additional evaluations increase the total runtime compared to a standard motion generation loop.

Outputs:
  results.txt       -> pose(3) + current_lengths(3) (6 cols)
  results_head.csv  -> full logs with header, now with L_fk1..L_fk3
Units: mm
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn

# --------- Timing defaults ----------
ITER_DEF = 1000
DT_DEF   = 0.004
SUBMV_T  = 1.2

# --------- Control defaults ----------
KP_DEF       = 100.0                 # isotropic gain in length-space
BQ_DIAG_DEF  = [0.08, 0.08, 0.08]  # pose-side damping (xyz)
LAM2_DEF     = 1e-4
POSE0_DEF    = [0.0, 0.0, -300.0]   # initial xyz [mm]

# ---------------- Helpers ----------------
def min_jerk_s(t, T):
    tau = np.clip(t / max(T, 1e-9), 0.0, 1.0)
    return 10*tau**3 - 15*tau**4 + 6*tau**5

def dls_pinv(J, lam2=1e-4):
    # J: (3,3); returns J^+ (3,3)
    JT = J.T
    I  = np.eye(J.shape[0])
    return JT @ np.linalg.inv(J @ JT + lam2 * I)

def model_device(model: nn.Module):
    for p in model.parameters():
        return p.device
    return torch.device('cpu')

# ---------------- Geometry (match pinntrain_delta.py) ----------------
def load_geometry_KR3D1200():
    """
    Geometry simplified as 3-RPS Delta configuration.
    Base radius Rb, platform radius Rp, anchors at 90/210/330 deg.
    Platform is translation-only (no rotation) in this controller.
    """
    Rb, Rp = 280.0, 80.0        # [mm]
    theta = np.deg2rad([0, 120, 240])
    z_base = 0.0

    base_points = np.stack([
        Rb*np.cos(theta),
        Rb*np.sin(theta),
        np.full_like(theta, z_base)
    ], axis=1)                   # (3,3)

    platform_points = np.stack([
        Rp*np.cos(theta),
        Rp*np.sin(theta),
        np.zeros_like(theta)
    ], axis=1)                   # (3,3)

    return base_points.astype(np.float64), platform_points.astype(np.float64)

_B_np, _P_np = load_geometry_KR3D1200()  # np (3,3)
_B_t = torch.tensor(_B_np, dtype=torch.float32)
_P_t = torch.tensor(_P_np, dtype=torch.float32)
def fk_lengths_from_pose(pose_xyz):
    # torch
    if isinstance(pose_xyz, torch.Tensor):
        dev  = pose_xyz.device
        dt   = pose_xyz.dtype
        B = _B_t.to(device=dev, dtype=dt)   # (3,3)
        P = _P_t.to(device=dev, dtype=dt)   # (3,3)
        Pw = P + pose_xyz.reshape(1, 3)     # (3,3)
        L  = torch.linalg.norm(B - Pw, dim=1)  # torch[3]
        return L

    # numpy 分支
    pose_xyz = np.asarray(pose_xyz, dtype=np.float64).reshape(1, 3)
    Pw = _P_np + pose_xyz   # (3,3)
    L  = np.linalg.norm(_B_np - Pw, axis=1)  # np[3]
    return L

# ---------------- Model shell ----------------
class DeltaPINN(nn.Module):
    """
    pose_batch[B,3] --> Pw[B,3,3] --> x[B,9] --fc_output--> L[B,3]
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.register_buffer(
            "platform_points_local",
            torch.tensor([[ 80.0,   0.0, 0.0],
                          [-40.0,  69.28, 0.0],
                          [-40.0, -69.28, 0.0]], dtype=torch.float32)
        )
        self.fc_output = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softplus()
        )

    def forward(self, pose_batch):  # [B,3] -> [B,3]
        t = pose_batch[:, :3]                                 # [B,3]
        Pw = self.platform_points_local.unsqueeze(0) + t.unsqueeze(1)  # [B,3,3]
        x  = Pw.reshape(Pw.shape[0], -1)                      # [B,9]
        return self.fc_output(x)                              # [B,3]

    @torch.no_grad()
    def predict_lengths(self, pose_vec):   # pose_vec: [3] tensor (device=net)
        return self.forward(pose_vec.unsqueeze(0))[0]

    def get_jacobian(self, pose_vec):      # -> [3,3] numpy
        x = pose_vec.detach().clone().requires_grad_(True).unsqueeze(0)  # [1,3]
        L = self.forward(x)  # [1,3]
        rows = []
        for i in range(3):
            grad_i = torch.autograd.grad(L[0, i], x, retain_graph=True, create_graph=False)[0][0]
            rows.append(grad_i)
        J = torch.stack(rows, dim=0)  # [3,3]
        return J.detach().cpu().numpy()

def _metrics_setpoint(e_set):
    abs_e = np.abs(e_set)
    rmse  = float(np.sqrt(np.mean(e_set**2)))
    mae   = float(np.mean(abs_e))
    mabs  = float(np.max(abs_e))
    l2    = float(np.linalg.norm(e_set))
    return rmse, mae, mabs, l2

def _append_row_to_table_delta(row, fname="setpoint_table_delta.csv"):
    cols = [
        "tar_L1","tar_L2","tar_L3",
        "ik_L1","ik_L2","ik_L3",
        "e_set1","e_set2","e_set3",
        "RMSE","MAE","MaxAbs","L2","condJ",
        "x","y","z",
        "mapping","kp_or_vec","lam2","bq_diag",
        "dt","steps","submv_T","traj","model"
    ]
    vals = []
    vals += list(row["tar_L"])
    vals += list(row["ik_L"])
    vals += list(row["e_set"])
    vals += [row["RMSE"], row["MAE"], row["MaxAbs"], row["L2"], row["condJ"]]
    vals += list(row["pose"])
    vals += [
        row["mapping"], row["kp_or_vec"], row["lam2"],
        " ".join(map(str, row["bq_diag"])),
        row["dt"], row["steps"], row["submv_T"], row["traj"], row["model"]
    ]
    need_header = not os.path.exists(fname)
    with open(fname, "a", encoding="utf-8") as f:
        if need_header: f.write(",".join(cols) + "\n")
        f.write(",".join(str(v) for v in vals) + "\n")

# ---------------- Core step ----------------
def core_step(pose_tensor, L_ref, model, args):
    """
    pose_tensor: torch[3] (net device)
    L_ref: np[3]
    returns: next_pose_tensor, L_cur(np3), F_task(np3), qdot(np3)
    """
    device = pose_tensor.device

    # Current lengths from model
    with torch.no_grad():
        L_cur_t = fk_lengths_from_pose(pose_tensor)      # torch[6]
        L_cur   = L_cur_t.detach().cpu().numpy()           # np[6]

    # Length-space P control
    Kp = np.eye(3) * float(args.kp)
    if args.kp_vec is not None and len(args.kp_vec) == 3:
        Kp = np.diag(np.array(args.kp_vec, dtype=float))
    e_L = np.asarray(L_ref, dtype=float) - L_cur
    F_task = Kp @ e_L  # length-space "force"

    # Mapping to pose-rate
    J = model.get_jacobian(pose_tensor)  # [3,3]
    if args.use_jt:
        qdot_task = J.T @ F_task
    else:
        Jp = dls_pinv(J, args.lam2)
        qdot_task = Jp @ F_task

    # Pose-side damping (xyz)
    BQ = np.diag(np.array(args.bq_diag, dtype=float))
    qdot = (np.eye(3) - BQ) @ qdot_task

    # Euler update
    pose_next = pose_tensor + torch.as_tensor(qdot * args.dt, dtype=pose_tensor.dtype, device=device)
    return pose_next, L_cur, F_task, qdot

# ---------------- Runner ----------------
def run_controller(model, args, target_lengths):
    device = model_device(model)
    pose = torch.tensor(np.array(args.pose0, dtype=np.float32), device=device)

    # Initial lengths
    with torch.no_grad():
        L0_t = fk_lengths_from_pose(pose)      # torch[6]
        L0 = L0_t.detach().cpu().numpy()           # np[6]

    logs = []  # [t, L_cur(3), L_ref(3), F(3), qdot(3), pose(3), L_fk(3)]
    for k in range(args.steps):
        t = k * args.dt
        s = min_jerk_s(t, args.submv_T)
        L_ref = L0 + s * (np.asarray(target_lengths, dtype=float) - L0)

        pose, L_cur, F_task, qdot = core_step(pose, L_ref, model, args)
        pose_np = pose.detach().cpu().numpy()
        L_fk = fk_lengths_from_pose(pose_np)  # FK check at each step (cheap)

        logs.append([t,
                     *L_cur.tolist(),
                     *L_ref.tolist(),
                     *F_task.tolist(),
                     *qdot.tolist(),
                     *pose_np.tolist(),
                     *L_fk.tolist()])

    # Save logs
    arr = np.asarray(logs, dtype=float)

    # results.txt: pose(3)+L(3) from model
    basic = np.hstack([arr[:, -6:-3], arr[:, 1:4]])  # x,y,z | L1,L2,L3(model)
    np.savetxt("results_d.txt", basic, fmt="%.6f")

    # CSV with header (now includes L_fk1..L_fk3)
    header = "time,L1,L2,L3,Lref1,Lref2,Lref3,F1,F2,F3,qdot1,qdot2,qdot3,x,y,z,L_fk1,L_fk2,L_fk3"
    np.savetxt("results_head.csv", arr, fmt="%.6f", header=header, comments="")

    # Return last states for printing
    final_pose = arr[-1, -6:-3]        # x,y,z
    final_L_model = arr[-1, 1:4]       # L1..L3 (model-pred at last step)
    final_L_fk    = arr[-1, -3:]       # L_fk1..L_fk3 (FK at last step)

    target = np.asarray(target_lengths, dtype=float)
    e_set  = final_L_fk - target
    rmse, mae, mabs, l2 = _metrics_setpoint(e_set)

    try:
        J = model.get_jacobian(torch.tensor(final_pose, dtype=torch.float32, device=device))
        condJ = float(np.linalg.cond(J))
    except Exception:
        condJ = float("inf")

    e_str = " ".join(f"{v:+.4f}" for v in e_set)
    print("[info] --- Setpoint IK check (Delta) ---")
    print(f"[info] IK leg lengths (mm): {np.array2string(final_L_fk, precision=4, separator=', ')}")
    print(f"[info] e_set = IK - target (mm): [{e_str}]")
    print(f"[info] Metrics: RMSE={rmse:.4f}  MAE={mae:.4f}  MaxAbs={mabs:.4f}  L2={l2:.4f}  cond(J)={condJ:.2e}")

    mapping   = "JT" if args.use_jt else "DLS"
    kp_or_vec = ("vec" if args.kp_vec is not None else f"{args.kp}")
    row = {
        "tar_L": target.astype(float),
        "ik_L":  final_L_fk.astype(float),
        "e_set": e_set.astype(float),
        "RMSE": rmse, "MAE": mae, "MaxAbs": mabs, "L2": l2, "condJ": condJ,
        "pose": final_pose.astype(float),
        "mapping": mapping, "kp_or_vec": kp_or_vec, "lam2": float(args.lam2),
        "bq_diag": list(args.bq_diag),
        "dt": float(args.dt), "steps": int(args.steps), "submv_T": float(args.submv_T),
        "traj": args.traj, "model": args.model
    }
    _append_row_to_table_delta(row, fname="setpoint_table_delta.csv")

    print("[info] Files saved: results_d.txt, results_head.csv, setpoint_table_delta.csv")

    return final_pose, final_L_model, final_L_fk

# ---------------- CLI ----------------
def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="best_model_delta.pth")
    ap.add_argument("--target", type=float, nargs=3, default=[551.37, 531.49, 562.62], # put your target length here
                    help="target leg lengths [mm] for 3 legs")
    ap.add_argument("--pose0", type=float, nargs=3, default=POSE0_DEF,
                    help="initial xyz [mm]")
    ap.add_argument("--steps", type=int, default=ITER_DEF)
    ap.add_argument("--dt", type=float, default=DT_DEF)
    ap.add_argument("--submv-T", dest="submv_T", type=float, default=SUBMV_T)
    ap.add_argument("--traj", choices=["minjerk"], default="minjerk")  # only min-jerk here
    ap.add_argument("--sense", choices=["nn","ik"], default="ik",
        help="how to measure current leg lengths at each step: nn (model forward) or ik (geometry)") # As the model is mainly used for abtaining Jac, we recommanded using IK here
    # Control
    ap.add_argument("--kp", type=float, default=KP_DEF)
    ap.add_argument("--kp-vec", type=float, nargs=3, help="per-leg gains (3)")
    ap.add_argument("--use-jt", action="store_true", help="use J^T (default: DLS)")
    ap.add_argument("--lam2", type=float, default=LAM2_DEF)
    ap.add_argument("--bq-diag", type=float, nargs=3, default=BQ_DIAG_DEF)
    return ap

def main():
    args = build_arg_parser().parse_args()

    # Load model (must match pinntrain_delta.py topology)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DeltaPINN().to(device)
    sd = torch.load(args.model, map_location="cpu")
    net.load_state_dict(sd)
    net.eval()

    final_pose, L_model, L_fk = run_controller(net, args, np.array(args.target, dtype=float))

    # Print comparison
    diff = L_model - L_fk
    rmse = np.sqrt(np.mean(diff**2))
    print("[info] done.")
    print(f"[info] final pose (mm)     = {final_pose}")
    print(f"[info] lengths (model)     = {L_model}")
    print(f"[info] lengths (FK check)  = {L_fk}")
    print(f"[info] per-leg error (mm)  = {diff}   (model - FK)")
    print(f"[info] RMSE (mm)           = {rmse:.4f}")

if __name__ == "__main__":
    main()
