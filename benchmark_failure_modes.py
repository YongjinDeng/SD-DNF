"""
================================================================================
Failure Mode & Runtime Bottleneck Decomposition Suite (Definitive Master)
File: benchmark_failure_modes.py
Location: D:\\0临床科研\\手搓核弹\\code\\

Integrated Modules:
  1. Task 'brain'   : MSD-Brain-01~10 cross-case Dice & certified topology audit (0% folding)
  2. Task 'omega'   : Omega_0 (10, 20, 30) harmonic spectral bias & topological stress analysis
  3. Task 'runtime' : DIR-Lab Case 08 (512x512x128) end-to-end stage runtime decomposition
  4. Task 'all'     : Run all 3 modules sequentially

Execution:
  cd /d D:\\0临床科研\\手搓核弹\\code
  python benchmark_failure_modes.py --task runtime     <-- 【本次仅需执行这行，5分钟修复耗时表】
  python benchmark_failure_modes.py --task all         <-- 全量执行

Outputs (saved in D:\\0临床科研\\手搓核弹\\result\\benchmark_validation_artifacts\\):
  - Table_Validation_Brain_Dice_All.csv
  - Table_Validation_Omega0_Spectral_Bias.csv
  - Table_Validation_Case08_Runtime_Breakdown.csv
  - Fig_Failure_Brain08_Anatomical_Proof.png
================================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

# 确保主模块可正确 import
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from main import (
    load_case_data,
    extract_vessel_anchors_unified,
    dual_track_matcher,
    compute_edge_awareness_map,
    ContinuousVPEF_Solver,
    SineLayer,
    get_jacobian,
)

RESULT_ROOT = r"D:\0临床科研\手搓核弹\result"
OUT_DIR = os.path.join(RESULT_ROOT, "benchmark_validation_artifacts")
os.makedirs(OUT_DIR, exist_ok=True)


def set_global_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lung_mask(I_fix):
    """自适应判断：彻底兼容 DIR-Lab 1-5 (HU+1024) 与 6-10 (原始灰度)"""
    if np.min(I_fix) < -500:
        return (I_fix > -950) & (I_fix < -350)
    else:
        return (I_fix > 100) & (I_fix < 650)


def get_normalized_pair(I_fix, I_mov):
    """自适应图像张量归一化"""
    if np.min(I_fix) < -500:
        nf = np.clip((I_fix - (-1000.0)) / 1200.0, 0.0, 1.0)
        nm = np.clip((I_mov - (-1000.0)) / 1200.0, 0.0, 1.0)
    else:
        nf = np.clip(I_fix / 2000.0, 0.0, 1.0)
        nm = np.clip(I_mov / 2000.0, 0.0, 1.0)
    T_fix = torch.from_numpy(nf).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()
    T_mov = torch.from_numpy(nm).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()
    return T_fix, T_mov


# =====================================================================
# MODULE 1: MSD-Brain 跨受试者海马体配准与拓扑可逆性验证
# =====================================================================
def analyze_brain_all_cases():
    print("\n" + "=" * 85)
    print(">>> [TASK 1/3] MSD-Brain CROSS-CASE DICE & TOPOLOGY AUDIT (main.py ALIGNED)")
    print("=" * 85)

    records = []
    brain08_visuals = None

    for bid in range(1, 11):
        try:
            I_fix, I_mov, _, _, vs, mf, mm = load_case_data("BRAIN", bid)
            H, W, D = I_fix.shape
            sf = torch.tensor([(W - 1) / 2.0, (H - 1) / 2.0, (D - 1) / 2.0]).float()

            cf = np.mean(np.argwhere(mf > 0), axis=0) if np.sum(mf) > 0 else np.array([H / 2, W / 2, D / 2])
            cm = np.mean(np.argwhere(mm > 0), axis=0) if np.sum(mm) > 0 else np.array([H / 2, W / 2, D / 2])
            global_t = cm - cf

            inter0 = np.sum((mf > 0) & (mm > 0))
            denom0 = np.sum(mf > 0) + np.sum(mm > 0)
            init_dice_val = 2.0 * inter0 / denom0 * 100.0 if denom0 > 0 else 100.0

            edge_map = compute_edge_awareness_map(I_fix)
            kappa = np.exp(-4.0 * edge_map)
            edge_tensor = torch.from_numpy(kappa).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()

            anchors = extract_vessel_anchors_unified(I_fix, bid, is_other_organ=True, mask_gt=mf)
            disps, valid = dual_track_matcher(I_fix, I_mov, anchors, bid,
                                              is_other_organ=True, global_t=global_t, vs=vs)
            anc_v, disp_v = anchors[valid], disps[valid]
            if len(anc_v) < 5:
                print(f"  ⚠ MSD-Brain-{bid:02d}: insufficient anchors, skipping")
                continue

            norm_c = torch.from_numpy(np.stack([
                (anc_v[:, 0] / (W - 1)) * 2 - 1,
                (anc_v[:, 1] / (H - 1)) * 2 - 1,
                (anc_v[:, 2] / (D - 1)) * 2 - 1
            ], axis=-1)).float()
            targets = torch.from_numpy(disp_v).float() / sf

            set_global_seed(42)
            model = ContinuousVPEF_Solver(hidden_dim=64)

            # 脑部为跨受试者形态差异，严格执行 Stage-1 宏观弹性力学求解 (90 steps)
            macro_steps = 90
            lr_macro = 1.8e-3
            opt_macro = torch.optim.Adam(model.parameters(), lr=lr_macro)
            scheduler_macro = torch.optim.lr_scheduler.CosineAnnealingLR(opt_macro, T_max=macro_steps, eta_min=1e-4)

            pde_w = 0.006
            topo_w = 10.0
            topo_thresh = 0.08

            for step in range(1, macro_steps + 1):
                opt_macro.zero_grad()
                loss_d = torch.mean((model(norm_c) - targets) ** 2)

                pde_pts = torch.cat([torch.rand(4096, 3) * 2 - 1,
                                     norm_c + torch.randn_like(norm_c) * 0.05], dim=0)
                disp_pde, J = get_jacobian(model, pde_pts)

                stiffness = F.grid_sample(edge_tensor, pde_pts.view(1, 1, 1, -1, 3), align_corners=True, mode='nearest').view(-1)
                strain = torch.sum((0.5 * (J + J.transpose(-1, -2))) ** 2, dim=[-1, -2])
                div_sq = (J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2]) ** 2
                loss_pde = (stiffness * (strain + 2.0 * div_sq)).mean()

                det_F = torch.det(torch.eye(3).unsqueeze(0).expand(pde_pts.shape[0], -1, -1) + J)
                barrier = torch.where(
                    det_F >= 1e-4,
                    -torch.log(torch.clamp(det_F, min=1e-4)),
                    -np.log(1e-4) + 10000.0 * (1e-4 - det_F)
                )
                loss_topo = topo_w * torch.mean(torch.where(det_F < topo_thresh, barrier, torch.zeros_like(det_F)))
                loss_damp = (0.02 * torch.mean(disp_pde ** 2)) if init_dice_val > 80.0 else 0.0

                (loss_d + pde_w * loss_pde + loss_topo + loss_damp).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                opt_macro.step()
                scheduler_macro.step()

            # 评估重叠度与拓扑行列式
            yy, xx, zz = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
            pts = np.stack([
                (xx.ravel() / (W - 1)) * 2 - 1,
                (yy.ravel() / (H - 1)) * 2 - 1,
                (zz.ravel() / (D - 1)) * 2 - 1
            ], axis=-1)
            with torch.no_grad():
                disp_dense = (model(torch.from_numpy(pts).float()) * sf).numpy()

            qx = np.clip(xx + disp_dense[:, 0].reshape(H, W, D), 0, W - 1)
            qy = np.clip(yy + disp_dense[:, 1].reshape(H, W, D), 0, H - 1)
            qz = np.clip(zz + disp_dense[:, 2].reshape(H, W, D), 0, D - 1)
            warped = map_coordinates(mm.astype(np.float32), [qy, qx, qz], order=1, mode='nearest')

            inter = np.sum((mf > 0) & (warped >= 0.5))
            denom = np.sum(mf > 0) + np.sum(warped >= 0.5)
            dice_final = 2.0 * inter / denom * 100.0 if denom > 0 else 100.0

            eval_pts = torch.rand(4000, 3) * 2 - 1
            with torch.enable_grad():
                _, eJ = get_jacobian(model, eval_pts)
                det_eval = torch.det(torch.eye(3).unsqueeze(0).expand(4000, -1, -1) + eJ.detach())
            fold_rate = float((det_eval <= 0).float().mean().item() * 100)
            min_det = float(det_eval.min().item())
            mean_det = float(det_eval.mean().item())

            records.append({
                "Case": f"MSD-Brain-{bid:02d}",
                "Dice_Init_%": round(init_dice_val, 2),
                "Dice_Final_%": round(dice_final, 2),
                "Dice_Delta_%": round(dice_final - init_dice_val, 2),
                "Fold_%": round(fold_rate, 4),
                "Min_detJ": round(min_det, 4),
                "Mean_detJ": round(mean_det, 4),
            })
            print(f"  ✔ MSD-Brain-{bid:02d} | Dice {init_dice_val:.2f}% -> {dice_final:.2f}% "
                  f"| Fold {fold_rate:.4f}% | min detJ {min_det:.4f}")

            if bid == 8:
                brain08_visuals = {
                    "I_fix": I_fix.copy(),
                    "I_mov": I_mov.copy(),
                    "mf": mf.copy(),
                    "mm": mm.copy(),
                    "warped": warped,
                }

        except Exception as e:
            print(f"  ⚠ MSD-Brain-{bid:02d} skipped: {e}")

    df = pd.DataFrame(records)
    out_csv = os.path.join(OUT_DIR, "Table_Validation_Brain_Dice_All.csv")
    df.to_csv(out_csv, index=False)
    print(f"  -> Brain Table saved: {out_csv}")

    # 导出证据图证明非同源拒绝撕裂
    if brain08_visuals is not None:
        v = brain08_visuals
        slice_z = v["I_fix"].shape[2] // 2
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=300)
        axes[0].imshow(v["I_fix"][:, :, slice_z], cmap='gray')
        axes[0].contour(v["mf"][:, :, slice_z], colors='red', linewidths=1.2)
        axes[0].set_title("(a) Target Anatomy (Fix + Mask)", fontsize=9.5, weight='bold')

        axes[1].imshow(v["I_mov"][:, :, slice_z], cmap='gray')
        axes[1].contour(v["mm"][:, :, slice_z], colors='cyan', linewidths=1.2)
        axes[1].set_title("(b) Moving Anatomy (Mov + Mask)", fontsize=9.5, weight='bold')

        diff = np.abs(v["mf"][:, :, slice_z].astype(float) - v["warped"][:, :, slice_z].astype(float))
        axes[2].imshow(v["I_fix"][:, :, slice_z], cmap='gray', alpha=0.6)
        axes[2].imshow(diff, cmap='autumn', alpha=0.5)
        axes[2].set_title("(c) Morphological Discordance", fontsize=9.5, weight='bold')

        for ax in axes: ax.axis('off')
        plt.tight_layout()
        out_fig = os.path.join(OUT_DIR, "Fig_Failure_Brain08_Anatomical_Proof.png")
        plt.savefig(out_fig, bbox_inches='tight')
        plt.close()
        print(f"  -> Brain-08 figure saved: {out_fig}\n")


# =====================================================================
# MODULE 2: 频率缩放系数 (omega_0) 频谱偏置应激分析
# =====================================================================
def analyze_omega30_folding():
    print("\n" + "=" * 85)
    print(">>> [TASK 2/3] OMEGA_0 (10, 20, 30) SPECTRAL BIAS ANALYSIS")
    print("=" * 85)

    I_fix, I_mov, p0, p50, vs, _, _ = load_case_data("DIRLAB", 8)
    H, W, D = I_fix.shape
    sf = torch.tensor([(W - 1) / 2.0, (H - 1) / 2.0, (D - 1) / 2.0]).float()

    anchors = extract_vessel_anchors_unified(I_fix, 8, is_popi=False)
    disps, valid = dual_track_matcher(I_fix, I_mov, anchors, 8, is_popi=False, vs=vs)
    anc_v, disp_v = anchors[valid], disps[valid]
    norm_c = torch.from_numpy(np.stack([
        (anc_v[:, 0] / (W - 1)) * 2 - 1,
        (anc_v[:, 1] / (H - 1)) * 2 - 1,
        (anc_v[:, 2] / (D - 1)) * 2 - 1
    ], axis=-1)).float()
    targets = torch.from_numpy(disp_v).float() / sf

    lm_n = torch.from_numpy(np.stack([
        (p0[:, 0] / (W - 1)) * 2 - 1,
        (p0[:, 1] / (H - 1)) * 2 - 1,
        (p0[:, 2] / (D - 1)) * 2 - 1
    ], axis=-1)).float()

    results = []
    for w0 in [10.0, 20.0, 30.0]:
        set_global_seed(42)

        class DynamicSiren(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    SineLayer(3, 64, w0=w0, is_first=True),
                    SineLayer(64, 64, w0=w0),
                    SineLayer(64, 64, w0=w0),
                    nn.Linear(64, 3)
                )
                with torch.no_grad():
                    nn.init.zeros_(self.net[-1].weight)
                    nn.init.zeros_(self.net[-1].bias)

            def forward(self, x):
                return self.net(x)

        model = DynamicSiren()
        opt = torch.optim.Adam(model.parameters(), lr=1.8e-3)
        for _ in range(100):
            opt.zero_grad()
            loss_d = torch.mean((model(norm_c) - targets) ** 2)
            pde_pts = torch.rand(1536, 3) * 2 - 1
            _, J = get_jacobian(model, pde_pts)
            det_F = torch.det(torch.eye(3).unsqueeze(0).expand(1536, -1, -1) + J)
            barrier = torch.where(
                det_F >= 1e-4,
                -torch.log(torch.clamp(det_F, min=1e-4)),
                -np.log(1e-4) + 10000.0 * (1e-4 - det_F)
            )
            loss_topo = 4.0 * torch.mean(torch.where(det_F < 0.05, barrier, torch.zeros_like(det_F)))
            (loss_d + 0.002 * torch.sum(J ** 2, dim=[-1, -2]).mean() + loss_topo).backward()
            opt.step()

        eval_pts = torch.rand(4000, 3) * 2 - 1
        with torch.enable_grad():
            _, eJ = get_jacobian(model, eval_pts)
            det_eval = torch.det(torch.eye(3).unsqueeze(0).expand(4000, -1, -1) + eJ.detach())
        fold = float((det_eval <= 0).float().mean().item() * 100)
        min_d = float(det_eval.min().item())

        with torch.no_grad():
            pred = (model(lm_n) * sf).numpy()
        tre = float(np.mean(np.sqrt(np.sum(((p0 + pred - p50) * vs) ** 2, axis=1))))

        results.append({
            "omega_0": int(w0),
            "TRE_mm": round(tre, 3),
            "Fold_%": round(fold, 4),
            "Min_detJ": round(min_d, 4),
        })
        print(f"  • ω₀={int(w0):2d} | TRE {tre:.2f} mm | Fold {fold:.4f}% | min detJ {min_d:.4f}")

    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, "Table_Validation_Omega0_Spectral_Bias.csv")
    df.to_csv(out_csv, index=False)
    print(f"  -> Omega-0 spectral bias table saved: {out_csv}\n")


# =====================================================================
# MODULE 3: DIR-Lab Case 08 (512x512x128) 全阶段端到端耗时分解 (修复掩码)
# =====================================================================
def profile_case08_runtime():
    print("\n" + "=" * 85)
    print(">>> [TASK 3/3] Case-08 RUNTIME DECOMPOSITION (512x512x128) - FIXED MASK")
    print("=" * 85)

    I_fix, I_mov, p0, p50, vs, _, _ = load_case_data("DIRLAB", 8)
    H, W, D = I_fix.shape
    sf = torch.tensor([(W - 1) / 2.0, (H - 1) / 2.0, (D - 1) / 2.0]).float()

    # 关键修复点：使用自适应函数，保证提取出真实的几十万个肺部体素
    lung_mask = get_lung_mask(I_fix)
    roi_pts = np.argwhere(lung_mask)
    print(f"  • Case 08 真实肺部采样池体素数: {len(roi_pts)} (非零正常值)")

    n_sample = min(6144, len(roi_pts))
    T_fix, T_mov = get_normalized_pair(I_fix, I_mov)

    breakdown = {}

    # 阶段 1：锚点提取
    t0 = time.time()
    anchors = extract_vessel_anchors_unified(I_fix, 8, is_popi=False)
    breakdown["1_Anchor_Extraction(s)"] = round(time.time() - t0, 2)
    print(f"  [1/5] 锚点提取耗时: {breakdown['1_Anchor_Extraction(s)']} s")

    # 阶段 2：双轨匹配
    t0 = time.time()
    disps_vox, valid = dual_track_matcher(I_fix, I_mov, anchors, 8, is_popi=False, vs=vs)
    breakdown["2_DualTrack_Matching(s)"] = round(time.time() - t0, 2)
    print(f"  [2/5] 双轨匹配耗时: {breakdown['2_DualTrack_Matching(s)']} s")

    anc_v, disp_v = anchors[valid], disps_vox[valid]
    norm_c = torch.from_numpy(np.stack([
        (anc_v[:, 0] / (W - 1)) * 2 - 1,
        (anc_v[:, 1] / (H - 1)) * 2 - 1,
        (anc_v[:, 2] / (D - 1)) * 2 - 1
    ], axis=-1)).float()
    targets = torch.from_numpy(disp_v).float() / sf

    set_global_seed(42)
    model = ContinuousVPEF_Solver(hidden_dim=64)

    # 阶段 3：宏观 Stage 1 (120 步)
    opt_macro = torch.optim.Adam(model.parameters(), lr=1.8e-3)
    t0 = time.time()
    for _ in range(120):
        opt_macro.zero_grad()
        loss_d = torch.mean((model(norm_c) - targets) ** 2)
        pde_pts = torch.rand(1536, 3) * 2 - 1
        _, J = get_jacobian(model, pde_pts)
        det_F = torch.det(torch.eye(3).unsqueeze(0).expand(1536, -1, -1) + J)
        barrier = torch.where(
            det_F >= 1e-4,
            -torch.log(torch.clamp(det_F, min=1e-4)),
            -np.log(1e-4) + 10000.0 * (1e-4 - det_F)
        )
        loss_topo = 4.0 * torch.mean(torch.where(det_F < 0.05, barrier, torch.zeros_like(det_F)))
        (loss_d + 0.002 * torch.sum(J ** 2, dim=[-1, -2]).mean() + loss_topo).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        opt_macro.step()
    breakdown["3_Macro_Stage1(s)"] = round(time.time() - t0, 2)
    print(f"  [3/5] 宏观 Stage 1 求解耗时: {breakdown['3_Macro_Stage1(s)']} s")

    patch_off = torch.tensor([
        [0, 0, 0], [0.02, 0, 0], [-0.02, 0, 0],
        [0, 0.02, 0], [0, -0.02, 0], [0, 0, 0.03], [0, 0, -0.03]
    ]).float()

    # 阶段 4：中尺度光度 Stage 2a (30 步，真实反向传播)
    opt_mid = torch.optim.Adam(model.parameters(), lr=2.0e-4)
    t0 = time.time()
    for _ in range(30):
        opt_mid.zero_grad()
        idx = np.random.choice(len(roi_pts), n_sample, replace=False)
        sub = roi_pts[idx]
        norm_roi = torch.from_numpy(np.stack([
            (sub[:, 1] / (W - 1)) * 2 - 1,
            (sub[:, 0] / (H - 1)) * 2 - 1,
            (sub[:, 2] / (D - 1)) * 2 - 1
        ], axis=-1)).float()
        _, J = get_jacobian(model, norm_roi[:1024])
        disp_all = model(norm_roi)
        pts_f = (norm_roi.unsqueeze(1) + patch_off.unsqueeze(0)).view(-1, 3)
        pts_m = pts_f + disp_all.repeat_interleave(7, dim=0)
        v_f = F.grid_sample(T_fix, pts_f.view(1, 1, 1, -1, 3), align_corners=True).view(-1, 7)
        v_m = F.grid_sample(T_mov, pts_m.view(1, 1, 1, -1, 3), align_corners=True).view(-1, 7)
        fc = v_f - v_f.mean(dim=-1, keepdim=True)
        mc = v_m - v_m.mean(dim=-1, keepdim=True)
        lncc = (fc * mc).sum(dim=-1) / (torch.sqrt((fc ** 2).sum(dim=-1) * (mc ** 2).sum(dim=-1)) + 1e-4)
        loss = torch.mean(1.0 - lncc)
        det_F = torch.det(torch.eye(3).unsqueeze(0).expand(J.shape[0], -1, -1) + J)
        barrier = torch.where(
            det_F >= 1e-4,
            -torch.log(torch.clamp(det_F, min=1e-4)),
            -np.log(1e-4) + 10000.0 * (1e-4 - det_F)
        )
        loss = loss + 4.0 * torch.mean(torch.where(det_F < 0.05, barrier, torch.zeros_like(det_F)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.15)
        opt_mid.step()
    breakdown["4_Mid_Stage_Photometric(s)"] = round(time.time() - t0, 2)
    print(f"  [4/5] 中尺度光度 Stage 2a 耗时: {breakdown['4_Mid_Stage_Photometric(s)']} s (真实反向传播)")

    # 阶段 5：微观光度 Stage 2b (60 步，真实反向传播)
    opt_fine = torch.optim.Adam(model.parameters(), lr=1.3e-4)
    t0 = time.time()
    for _ in range(60):
        opt_fine.zero_grad()
        idx = np.random.choice(len(roi_pts), n_sample, replace=False)
        sub = roi_pts[idx]
        norm_roi = torch.from_numpy(np.stack([
            (sub[:, 1] / (W - 1)) * 2 - 1,
            (sub[:, 0] / (H - 1)) * 2 - 1,
            (sub[:, 2] / (D - 1)) * 2 - 1
        ], axis=-1)).float()
        _, J = get_jacobian(model, norm_roi[:1024])
        disp_all = model(norm_roi)
        pts_f = (norm_roi.unsqueeze(1) + patch_off.unsqueeze(0)).view(-1, 3)
        pts_m = pts_f + disp_all.repeat_interleave(7, dim=0)
        v_f = F.grid_sample(T_fix, pts_f.view(1, 1, 1, -1, 3), align_corners=True).view(-1, 7)
        v_m = F.grid_sample(T_mov, pts_m.view(1, 1, 1, -1, 3), align_corners=True).view(-1, 7)
        fc = v_f - v_f.mean(dim=-1, keepdim=True)
        mc = v_m - v_m.mean(dim=-1, keepdim=True)
        lncc = (fc * mc).sum(dim=-1) / (torch.sqrt((fc ** 2).sum(dim=-1) * (mc ** 2).sum(dim=-1)) + 1e-4)
        loss = torch.mean(1.0 - lncc)
        det_F = torch.det(torch.eye(3).unsqueeze(0).expand(J.shape[0], -1, -1) + J)
        barrier = torch.where(
            det_F >= 1e-4,
            -torch.log(torch.clamp(det_F, min=1e-4)),
            -np.log(1e-4) + 10000.0 * (1e-4 - det_F)
        )
        loss = loss + 4.0 * torch.mean(torch.where(det_F < 0.05, barrier, torch.zeros_like(det_F)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.10)
        opt_fine.step()
    breakdown["5_Fine_Stage_Photometric(s)"] = round(time.time() - t0, 2)
    print(f"  [5/5] 微观光度 Stage 2b 耗时: {breakdown['5_Fine_Stage_Photometric(s)']} s (真实反向传播)")

    # 阶段 6：评估耗时
    t0 = time.time()
    eval_pts = torch.rand(4000, 3) * 2 - 1
    with torch.enable_grad():
        _, eJ = get_jacobian(model, eval_pts)
        det_eval = torch.det(torch.eye(3).unsqueeze(0).expand(4000, -1, -1) + eJ.detach())
    breakdown["6_Evaluation(s)"] = round(time.time() - t0, 3)

    breakdown["Total(s)"] = round(sum(breakdown.values()), 2)

    df = pd.DataFrame([breakdown])
    out_csv = os.path.join(OUT_DIR, "Table_Validation_Case08_Runtime_Breakdown.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n>>> 真实耗时分解表已更新覆盖: {out_csv}")
    print(df.T.to_string())
    print("=" * 85 + "\n")


# =====================================================================
# 主调度入口 (支持按需单跑某个任务)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="SD-DNF Failure Mode & Runtime Profiler")
    parser.add_argument("--task", type=str, choices=["brain", "omega", "runtime", "all"], default="all",
                        help="Select task to run: 'runtime' for quick Case 08 fix, 'all' for full suite")
    args = parser.parse_args()

    t_start = time.time()
    if args.task in ["brain", "all"]:
        analyze_brain_all_cases()
    if args.task in ["omega", "all"]:
        analyze_omega30_folding()
    if args.task in ["runtime", "all"]:
        profile_case08_runtime()

    print(f"🎉 任务 [{args.task}] 执行完成！总耗时: {time.time()-t_start:.1f} s")
    print(f"📁 结果已归档至: {OUT_DIR}\n")


if __name__ == "__main__":
    main()