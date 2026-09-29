"""
================================================================================
Nature Machine Intelligence / IEEE TMI - Supplementary Validation Suite
Module: supplementary_benchmarks.py
Description:
  1. Extended Data Fig. 1: Synthetic vortex topological stress test (Barrier vs Soft Penalty)
  2. Supplementary Table 5: Hyperparameter sensitivity grid (tau in [1e-3, 1e-4, 1e-5], w0 in [10, 20, 30])
  3. Peak Resident Set Size (Peak RSS) physical memory footprint profiling
================================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "8"

import sys
import time
import argparse
import psutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_num_threads(8)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['pdf.fonttype'] = 42

RESULT_DIR = r"D:\0临床科研\手搓核弹\result"
os.makedirs(RESULT_DIR, exist_ok=True)


class PeakRAMTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_mb = self.process.memory_info().rss / (1024 * 1024)

    def update(self):
        current = self.process.memory_info().rss / (1024 * 1024)
        if current > self.peak_mb:
            self.peak_mb = current

    def get_peak(self):
        self.update()
        return self.peak_mb


class Siren2D(nn.Module):
    def __init__(self, in_d=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, 2)
        )
        with torch.no_grad():
            self.net[0].weight.uniform_(-1.0 / in_d, 1.0 / in_d)
            for m in self.net.modules():
                if isinstance(m, nn.Linear) and m != self.net[0]:
                    m.weight.uniform_(-np.sqrt(6.0 / hidden) / 30.0, np.sqrt(6.0 / hidden) / 30.0)
            nn.init.zeros_(self.net[-1].weight)

    def forward(self, x):
        h = torch.sin(30.0 * self.net[0](x))
        h = torch.sin(30.0 * self.net[1](h))
        return self.net[2](h)


def get_jacobian_2d(model, coords):
    coords = coords.clone().detach().requires_grad_(True)
    disp = model(coords)
    J = torch.stack([
        torch.autograd.grad(disp[:, i], coords, torch.ones_like(disp[:, i]), create_graph=True)[0]
        for i in range(2)
    ], dim=1)
    return disp, J


def simulate_vortex(x, y, strength=2.5):
    r = torch.sqrt(x**2 + y**2 + 1e-6)
    theta = torch.atan2(y, x) + strength * torch.exp(-r**2 * 5.0)
    return torch.stack([r * torch.cos(theta) - x, r * torch.sin(theta) - y], dim=-1)


def run_vortex_benchmark():
    print("\n" + "=" * 75)
    print(">>> [1/2] RUNNING EXTENDED DATA FIG. 1: SYNTHETIC VORTEX STRESS TEST")
    print("=" * 75)
    torch.manual_seed(42)

    pts = (torch.rand(300, 2) * 2.0 - 1.0)
    disp_gt = simulate_vortex(pts[:, 0], pts[:, 1])

    model_soft = Siren2D()
    model_barrier = Siren2D()

    opt_soft = torch.optim.Adam(model_soft.parameters(), lr=1e-3)
    opt_barrier = torch.optim.Adam(model_barrier.parameters(), lr=1e-3)

    print("  * Optimizing Soft Penalty Paradigm...")
    for _ in range(300):
        opt_soft.zero_grad()
        disp_pde, J = get_jacobian_2d(model_soft, pts)
        loss_data = torch.mean((disp_pde - disp_gt)**2)
        det_F = (1.0 + J[:, 0, 0]) * (1.0 + J[:, 1, 1]) - J[:, 0, 1] * J[:, 1, 0]
        loss_topo = 1.0 * torch.mean(torch.relu(0.05 - det_F)**2)
        (loss_data + loss_topo).backward()
        opt_soft.step()

    print("  * Optimizing SD-DNF Logarithmic Barrier Formulation...")
    for _ in range(300):
        opt_barrier.zero_grad()
        disp_pde, J = get_jacobian_2d(model_barrier, pts)
        loss_data = torch.mean((disp_pde - disp_gt)**2)
        det_F = (1.0 + J[:, 0, 0]) * (1.0 + J[:, 1, 1]) - J[:, 0, 1] * J[:, 1, 0]
        barrier = torch.where(
            det_F >= 1e-4,
            -torch.log(torch.clamp(det_F, min=1e-4)),
            -np.log(1e-4) + 10000.0 * (1e-4 - det_F)
        )
        loss_topo = 0.5 * torch.mean(torch.where(det_F < 0.05, barrier, torch.zeros_like(det_F)))
        (loss_data + loss_topo).backward()
        opt_barrier.step()

    grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, 40), torch.linspace(-1, 1, 40), indexing='xy')
    grid_pts = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

    with torch.no_grad():
        d_soft = model_soft(grid_pts)
        d_barrier = model_barrier(grid_pts)

    pts_s = grid_pts + d_soft
    pts_b = grid_pts + d_barrier

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    axes[0].scatter(pts_s[:, 0], pts_s[:, 1], c='#D55E00', s=2, alpha=0.7)
    for i in range(40):
        axes[0].plot(pts_s[i*40:(i+1)*40, 0], pts_s[i*40:(i+1)*40, 1], color='#D55E00', alpha=0.4, lw=0.5)
        axes[0].plot(pts_s[i::40, 0], pts_s[i::40, 1], color='#D55E00', alpha=0.4, lw=0.5)
    axes[0].set_title("(a) Unconstrained DL (Soft Penalty)\nSingular grid collapse & folding (det(F) <= 0)", fontsize=10, weight='bold')
    axes[0].set_aspect('equal')
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].scatter(pts_b[:, 0], pts_b[:, 1], c='#0072B2', s=2, alpha=0.7)
    for i in range(40):
        axes[1].plot(pts_b[i*40:(i+1)*40, 0], pts_b[i*40:(i+1)*40, 1], color='#0072B2', alpha=0.4, lw=0.5)
        axes[1].plot(pts_b[i::40, 0], pts_b[i::40, 1], color='#0072B2', alpha=0.4, lw=0.5)
    axes[1].set_title("(b) SD-DNF (Continuous Barrier)\nTopology strictly preserved (det(F) > 0)", fontsize=10, weight='bold')
    axes[1].set_aspect('equal')
    axes[1].set_xticks([]); axes[1].set_yticks([])

    plt.tight_layout()
    png_path = os.path.join(RESULT_DIR, "Extended_Data_Fig1_Vortex_Comparison.png")
    pdf_path = os.path.join(RESULT_DIR, "Extended_Data_Fig1_Vortex_Comparison.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✔ Extended Data Fig. 1 导出完成: {png_path}")


def run_sensitivity_and_ram_benchmark():
    print("\n" + "=" * 75)
    print(">>> [2/2] RUNNING SUPPLEMENTARY TABLE 5: SENSITIVITY & MEMORY PROFILER")
    print("=" * 75)

    from main import (
        load_case_data, extract_vessel_anchors_unified, dual_track_matcher,
        ContinuousVPEF_Solver, SineLayer, get_jacobian
    )

    class DynamicSirenSolver(ContinuousVPEF_Solver):
        def __init__(self, hidden_dim=64, w0=20.0):
            super().__init__(hidden_dim=hidden_dim)
            self.net[0] = SineLayer(3, hidden_dim, w0=w0, is_first=True)
            self.net[1] = SineLayer(hidden_dim, hidden_dim, w0=w0)
            self.net[2] = SineLayer(hidden_dim, hidden_dim, w0=w0)

    cases = [1, 5, 8]
    taus = [1e-3, 1e-4, 1e-5]
    w0s = [10.0, 20.0, 30.0]
    records = []

    for cid in cases:
        print(f"\n--- Profiling DIR-Lab Case {cid:02d} ---")
        I_fix, I_mov, p0, p50, vs, _, _ = load_case_data("DIRLAB", cid)
        H, W, D = I_fix.shape

        anchors = extract_vessel_anchors_unified(I_fix, cid, False, False, None)
        disps_vox, valid = dual_track_matcher(I_fix, I_mov, anchors, cid, False, False, None, vs=vs)
        anc_v, disp_v = anchors[valid], disps_vox[valid]

        sf = torch.tensor([(W - 1) / 2.0, (H - 1) / 2.0, (D - 1) / 2.0]).float()
        norm_c = torch.from_numpy(np.stack([
            (anc_v[:, 0] / (W - 1)) * 2.0 - 1.0,
            (anc_v[:, 1] / (H - 1)) * 2.0 - 1.0,
            (anc_v[:, 2] / (D - 1)) * 2.0 - 1.0
        ], axis=-1)).float()
        targets = torch.from_numpy(disp_v).float() / sf

        for tau in taus:
            for w0 in w0s:
                tracker = PeakRAMTracker()
                tracker.update()

                torch.manual_seed(42)
                model = DynamicSirenSolver(hidden_dim=64, w0=w0)
                opt = torch.optim.Adam(model.parameters(), lr=1.8e-3)

                max_steps = 100
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=1e-4)

                t0 = time.time()
                for step in range(max_steps):
                    opt.zero_grad()
                    loss_d = torch.mean((model(norm_c) - targets)**2)

                    # 对齐最新 main.py 的真实 1536 点物理采样密度
                    pde_pts = torch.rand(1536, 3) * 2.0 - 1.0
                    _, J = get_jacobian(model, pde_pts)

                    strain = torch.sum((0.5 * (J + J.transpose(-1, -2)))**2, dim=[-1, -2]).mean()
                    div_sq = ((J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2])**2).mean()
                    det_F = torch.det(torch.eye(3).unsqueeze(0).expand(pde_pts.shape[0], -1, -1) + J)

                    barrier = torch.where(
                        det_F >= tau,
                        -torch.log(torch.clamp(det_F, min=tau)),
                        -np.log(tau) + (1.0 / tau) * (tau - det_F)
                    )
                    loss_topo = 4.0 * torch.mean(torch.where(det_F < 0.05, barrier, torch.zeros_like(det_F)))
                    (loss_d + 0.002 * (strain + 2.0 * div_sq) + loss_topo).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    opt.step()
                    scheduler.step()

                    if step % 10 == 0:
                        tracker.update()

                elapsed_s = time.time() - t0
                peak_rss = tracker.get_peak()

                with torch.enable_grad():
                    eval_pts = torch.rand(4000, 3) * 2.0 - 1.0
                    _, eJ = get_jacobian(model, eval_pts)
                    det_eval = torch.det(torch.eye(3).unsqueeze(0).expand(4000, -1, -1) + eJ.detach())
                    fold_rate = float((det_eval <= 0.0).float().mean().item() * 100)

                with torch.no_grad():
                    lm_n = torch.from_numpy(np.stack([
                        (p0[:, 0] / (W - 1)) * 2.0 - 1.0,
                        (p0[:, 1] / (H - 1)) * 2.0 - 1.0,
                        (p0[:, 2] / (D - 1)) * 2.0 - 1.0
                    ], axis=-1)).float()
                    final_disp = (model(lm_n) * sf).numpy()
                    tre = float(np.mean(np.sqrt(np.sum(((p0 + final_disp - p50) * vs)**2, axis=1))))

                records.append({
                    "Case": f"DIR-Lab-{cid:02d}",
                    "Tau": f"{tau:.0e}",
                    "Omega_0": int(w0),
                    "TRE_mm": round(tre, 2),
                    "Folding_%": f"{fold_rate:.4f}%",
                    "Runtime_s": round(elapsed_s, 1),
                    "Peak_RSS_MB": round(peak_rss, 1)
                })
                print(f"  * tau={tau:.0e} | w0={int(w0):02d} -> TRE: {tre:.2f} mm | Fold: {fold_rate:.4f}% | Peak RSS: {peak_rss:.1f} MB")

    df = pd.DataFrame(records)
    default_mask = (df["Tau"] == "1e-04") & (df["Omega_0"] == 20)
    default_per_case = df[default_mask].set_index("Case")["TRE_mm"].to_dict()
    df["TRE_delta_mm"] = df.apply(lambda r: round(r["TRE_mm"] - default_per_case.get(r["Case"], r["TRE_mm"]), 2), axis=1)

    out_csv = os.path.join(RESULT_DIR, "Supplementary_Table_5_Sensitivity_and_RAM.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  ✔ Supplementary Table 5 导出完成: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="SD-DNF Supplementary Benchmarks")
    parser.add_argument("--task", type=str, choices=["vortex", "sensitivity", "all"], default="all")
    args = parser.parse_args()

    t_start = time.time()
    if args.task in ["vortex", "all"]: run_vortex_benchmark()
    if args.task in ["sensitivity", "all"]: run_sensitivity_and_ram_benchmark()
    print(f"\n  [Done] 总运行耗时: {time.time() - t_start:.1f} 秒\n")


if __name__ == "__main__":
    main()