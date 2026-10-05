"""
================================================================================
SD-DNF Benchmark Validation Suite (Autonomous Add-on) v3
File: benchmark_validation_suite.py
Location: D:\\0临床科研\\手搓核弹\\code\\

Modules:
  1. Task 'baseline'   : Budget-matched IDIR replication (SIREN + Bending + LNCC)
  2. Task 'ablation'   : Scale-decoupling ablation (aligned to main.py: 80+30+60=170)
  3. Task 'robustness' : Anchor density (relative ratio) & noise robustness
  4. Task 'stats'      : Holm-Bonferroni correction & effect sizes

Fixes over v2:
  - '--task all' skips baseline if output CSV already exists (use --force to override)
  - Unified HU/raw detection for lung_mask across all tasks
  - make_norm_roi_fn now takes explicit arguments (no closure hazard)
  - robustness supports multiple seeds (--seeds 3) to smooth the 60% dip
  - Global RNG/torch seed control for reproducibility

Execution:
  cd /d D:\\0临床科研\\手搓核弹\\code
  python benchmark_validation_suite.py --task ablation
  python benchmark_validation_suite.py --task robustness --seeds 3
  python benchmark_validation_suite.py --task all --force

Outputs:
  D:\\0临床科研\\手搓核弹\\result\\benchmark_validation_artifacts\\
    Table_Validation_Baseline.csv
    Table_Validation_Scale_Decoupling.csv
    Table_Validation_Anchor_Robustness.csv
    Table_Validation_Anchor_Aggregated.csv
    Table_Validation_Statistical_Correction.csv
================================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# =====================================================================
# Reproducibility helpers
# =====================================================================
def set_global_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lung_mask(I_fix):
    """Auto-detect HU vs raw and return a lung mask."""
    if np.min(I_fix) < -500:
        return (I_fix > -950) & (I_fix < -350)
    else:
        return (I_fix > 100) & (I_fix < 650)


def get_normalized_images(I_fix, I_mov):
    """Auto-detect HU vs raw and return [0,1]-normalized tensors."""
    if np.min(I_fix) < -500:
        nf = np.clip((I_fix - (-1000.0)) / 1200.0, 0.0, 1.0)
        nm = np.clip((I_mov - (-1000.0)) / 1200.0, 0.0, 1.0)
    else:
        nf = np.clip(I_fix / 2000.0, 0.0, 1.0)
        nm = np.clip(I_mov / 2000.0, 0.0, 1.0)
    H, W, D = I_fix.shape
    T_fix = torch.from_numpy(nf).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()
    T_mov = torch.from_numpy(nm).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()
    return T_fix, T_mov


# =====================================================================
# Shared evaluation utilities
# =====================================================================
def evaluate_topology(model, n_pts=4000, seed=42):
    g = torch.Generator().manual_seed(seed)
    with torch.enable_grad():
        eval_pts = torch.rand(n_pts, 3, generator=g) * 2 - 1
        _, eJ = get_jacobian(model, eval_pts)
        det_F = torch.det(torch.eye(3).unsqueeze(0).expand(n_pts, -1, -1) + eJ.detach())
        fold_rate = float((det_F <= 0.0).float().mean().item() * 100)
        min_det = float(det_F.min().item())
        mean_det = float(det_F.mean().item())
    return fold_rate, min_det, mean_det


def evaluate_tre(model, lm_n, sf, p0, p50, vs):
    with torch.no_grad():
        pred = (model(lm_n) * sf).numpy()
    return float(np.mean(np.sqrt(np.sum(((p0 + pred - p50) * vs) ** 2, axis=1))))


def build_norm_roi_fn(roi_pts, n_sample, pts_prob, W, H, D):
    """Explicit-argument version, no closure hazard."""
    def fn():
        if pts_prob is None:
            idx = np.random.choice(len(roi_pts), n_sample, replace=False)
        else:
            idx = np.random.choice(len(roi_pts), n_sample, replace=False, p=pts_prob)
        sub = roi_pts[idx]
        return torch.from_numpy(np.stack([
            (sub[:, 1] / (W - 1)) * 2 - 1,
            (sub[:, 0] / (H - 1)) * 2 - 1,
            (sub[:, 2] / (D - 1)) * 2 - 1
        ], axis=-1)).float()
    return fn


def compute_pts_prob(I_fix, lung_mask):
    """Gradient-weighted sampling probability over lung ROI."""
    dy, dx, dz = np.gradient(I_fix.astype(np.float32))
    ge = dx ** 2 + dy ** 2 + dz ** 2
    ge[~lung_mask] = 0.0
    pe = ge[lung_mask]
    pe = np.maximum(pe, 0)
    if np.sum(pe) <= 0:
        return None
    return (pe + 1e-4) / np.sum(pe + 1e-4)


def train_stage(model, norm_c, targets, edge_tensor, steps, lr,
                topo_w=4.0, pde_w=0.002, anchor_w=1.0,
                use_photometric=False, norm_roi_fn=None,
                T_fix=None, T_mov=None, patch_off=None):
    """Single training stage (anchor + PDE + barrier + optional photometric)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = anchor_w * torch.mean((model(norm_c) - targets) ** 2)

        pde_pts = torch.rand(1536, 3) * 2 - 1
        _, J = get_jacobian(model, pde_pts)
        stiffness = F.grid_sample(
            edge_tensor, pde_pts.view(1, 1, 1, -1, 3),
            align_corners=True, mode='nearest'
        ).view(-1)
        strain = torch.sum((0.5 * (J + J.transpose(-1, -2))) ** 2, dim=[-1, -2])
        div_sq = (J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2]) ** 2
        loss = loss + pde_w * (stiffness * (strain + 2.0 * div_sq)).mean()

        det_F = torch.det(torch.eye(3).unsqueeze(0).expand(pde_pts.shape[0], -1, -1) + J)
        barrier = torch.where(
            det_F >= 1e-4,
            -torch.log(torch.clamp(det_F, min=1e-4)),
            -np.log(1e-4) + 1e4 * (1e-4 - det_F)
        )
        loss = loss + topo_w * torch.mean(torch.where(det_F < 0.05, barrier, torch.zeros_like(det_F)))

        if use_photometric and norm_roi_fn is not None:
            norm_roi = norm_roi_fn()
            _, J_roi = get_jacobian(model, norm_roi[:1024])
            disp_all = model(norm_roi)
            pts_f = (norm_roi.unsqueeze(1) + patch_off.unsqueeze(0)).view(-1, 3)
            pts_m = pts_f + disp_all.repeat_interleave(7, dim=0)
            v_f = F.grid_sample(T_fix, pts_f.view(1, 1, 1, -1, 3), align_corners=True).view(-1, 7)
            v_m = F.grid_sample(T_mov, pts_m.view(1, 1, 1, -1, 3), align_corners=True).view(-1, 7)
            fc = v_f - v_f.mean(dim=-1, keepdim=True)
            mc = v_m - v_m.mean(dim=-1, keepdim=True)
            lncc = (fc * mc).sum(dim=-1) / (torch.sqrt((fc ** 2).sum(dim=-1) * (mc ** 2).sum(dim=-1)) + 1e-4)
            loss = loss + torch.mean(1.0 - lncc)

            det_F_roi = torch.det(torch.eye(3).unsqueeze(0).expand(J_roi.shape[0], -1, -1) + J_roi)
            barrier_roi = torch.where(
                det_F_roi >= 1e-4,
                -torch.log(torch.clamp(det_F_roi, min=1e-4)),
                -np.log(1e-4) + 1e4 * (1e-4 - det_F_roi)
            )
            loss = loss + topo_w * torch.mean(torch.where(det_F_roi < 0.05, barrier_roi, torch.zeros_like(det_F_roi)))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        opt.step()


# =====================================================================
# 1. Baseline: budget-matched IDIR replication
# =====================================================================
class IDIR_Continuous_Solver(nn.Module):
    def __init__(self, hidden_dim=64, w0=20.0):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(3, hidden_dim, w0=w0, is_first=True),
            SineLayer(hidden_dim, hidden_dim, w0=w0),
            SineLayer(hidden_dim, hidden_dim, w0=w0),
            nn.Linear(hidden_dim, 3)
        )
        with torch.no_grad():
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, coords):
        return self.net(coords)


def run_baseline_study(cases=None):
    if cases is None:
        cases = [1, 2, 3, 4, 5]

    print("\n" + "=" * 88)
    print(">>> [TASK 1/4] BASELINE: budget-matched single-resolution IDIR")
    print("=" * 88)
    records = []
    t_all = time.time()

    for cid in cases:
        t0 = time.time()
        I_fix, I_mov, p0, p50, vs, _, _ = load_case_data("DIRLAB", cid)
        H, W, D = I_fix.shape
        sf = torch.tensor([(W - 1) / 2.0, (H - 1) / 2.0, (D - 1) / 2.0]).float()

        lung_mask = get_lung_mask(I_fix)
        roi_pts = np.argwhere(lung_mask)
        if len(roi_pts) < 2048:
            print(f"  ⚠ [Case {cid:02d}] roi_pts too small ({len(roi_pts)}), skipping")
            continue

        T_fix, T_mov = get_normalized_images(I_fix, I_mov)

        patch_off = torch.tensor([
            [0, 0, 0], [0.02, 0, 0], [-0.02, 0, 0],
            [0, 0.02, 0], [0, -0.02, 0], [0, 0, 0.03], [0, 0, -0.03]
        ]).float()

        set_global_seed(42)
        model = IDIR_Continuous_Solver(hidden_dim=64, w0=20.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-4)

        for step in range(120):
            optimizer.zero_grad()
            idx = np.random.choice(len(roi_pts), min(4096, len(roi_pts)), replace=False)
            sub_pts = roi_pts[idx]
            norm_roi = torch.from_numpy(np.stack([
                (sub_pts[:, 1] / (W - 1)) * 2 - 1,
                (sub_pts[:, 0] / (H - 1)) * 2 - 1,
                (sub_pts[:, 2] / (D - 1)) * 2 - 1
            ], axis=-1)).float()

            disp, J = get_jacobian(model, norm_roi)
            loss_reg = 0.05 * torch.sum(J ** 2, dim=[-1, -2]).mean()

            pts_f = (norm_roi.unsqueeze(1) + patch_off.unsqueeze(0)).view(-1, 3)
            pts_m = pts_f + disp.repeat_interleave(7, dim=0)

            v_f = F.grid_sample(T_fix, pts_f.view(1, 1, 1, -1, 3), align_corners=True).view(-1, 7)
            v_m = F.grid_sample(T_mov, pts_m.view(1, 1, 1, -1, 3), align_corners=True).view(-1, 7)

            fc = v_f - v_f.mean(dim=-1, keepdim=True)
            mc = v_m - v_m.mean(dim=-1, keepdim=True)
            var_f = (fc ** 2).sum(dim=-1)
            var_m = (mc ** 2).sum(dim=-1)
            vt = (var_f > 1e-4) & (var_m > 1e-4)
            loss_photo = torch.mean(1.0 - ((fc * mc).sum(dim=-1) / torch.sqrt(var_f * var_m + 1e-4))[vt]) \
                if vt.sum() > 10 else torch.tensor(0.0)

            (loss_photo + loss_reg).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

        elapsed = time.time() - t0

        lm_n = torch.from_numpy(np.stack([
            (p0[:, 0] / (W - 1)) * 2 - 1,
            (p0[:, 1] / (H - 1)) * 2 - 1,
            (p0[:, 2] / (D - 1)) * 2 - 1
        ], axis=-1)).float()
        tre = evaluate_tre(model, lm_n, sf, p0, p50, vs)
        fold_rate, min_det, _ = evaluate_topology(model)

        print(f"  ✔ [Baseline] Case {cid:02d} | TRE: {tre:.2f} mm | Fold: {fold_rate:.4f}% "
              f"| Min det(F): {min_det:.4f} | Time: {elapsed:.1f}s")
        records.append({
            "Cohort": "DIRLAB",
            "Case": f"DIRLab-{cid:02d}",
            "Baseline_Replicated_TRE(mm)": round(tre, 2),
            "Baseline_Folding_Rate(%)": f"{fold_rate:.4f}%",
            "Baseline_Min_detF": round(min_det, 4),
            "Baseline_Runtime(s)": round(elapsed, 1),
            "Note": "Budget-matched single-resolution variant; not a full multi-resolution reimplementation."
        })

    df = pd.DataFrame(records)
    out_csv = os.path.join(OUT_DIR, "Table_Validation_Baseline.csv")
    df.to_csv(out_csv, index=False)
    print(f"  -> Baseline table saved: {out_csv}")
    print(f"  Total baseline time: {time.time()-t_all:.0f}s\n")
    return df


# =====================================================================
# 2. Ablation: scale-decoupling (aligned: 80+30+60=170 steps)
# =====================================================================
def run_scale_decoupling_ablation(cases=None):
    if cases is None:
        cases = [1, 2, 3, 4, 5]

    print("\n" + "=" * 88)
    print(">>> [TASK 2/4] SCALE-DECOUPLING ABLATION (aligned: 80+30+60=170 steps)")
    print("=" * 88)
    records = []
    t_all = time.time()

    for cid in cases:
        t_case = time.time()
        I_fix, I_mov, p0, p50, vs, _, _ = load_case_data("DIRLAB", cid)
        H, W, D = I_fix.shape
        sf = torch.tensor([(W - 1) / 2.0, (H - 1) / 2.0, (D - 1) / 2.0]).float()
        lm_n = torch.from_numpy(np.stack([
            (p0[:, 0] / (W - 1)) * 2 - 1,
            (p0[:, 1] / (H - 1)) * 2 - 1,
            (p0[:, 2] / (D - 1)) * 2 - 1
        ], axis=-1)).float()

        edge_map = compute_edge_awareness_map(I_fix)
        kappa = np.exp(-4.0 * edge_map)
        kappa[edge_map > np.percentile(edge_map, 95)] *= 0.5
        edge_tensor = torch.from_numpy(kappa).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()

        anchors = extract_vessel_anchors_unified(I_fix, cid, is_popi=False)
        disps_vox, valid = dual_track_matcher(I_fix, I_mov, anchors, cid, is_popi=False, vs=vs)
        anc_v, disp_v = anchors[valid], disps_vox[valid]
        norm_c = torch.from_numpy(np.stack([
            (anc_v[:, 0] / (W - 1)) * 2 - 1,
            (anc_v[:, 1] / (H - 1)) * 2 - 1,
            (anc_v[:, 2] / (D - 1)) * 2 - 1
        ], axis=-1)).float()
        targets = torch.from_numpy(disp_v).float() / sf

        lung_mask = get_lung_mask(I_fix)
        roi_pts = np.argwhere(lung_mask)
        if len(roi_pts) < 2048:
            print(f"  ⚠ [Case {cid:02d}] roi_pts too small ({len(roi_pts)}), skipping")
            continue
        n_sample = min(6144, len(roi_pts))
        pts_prob = compute_pts_prob(I_fix, lung_mask)
        T_fix, T_mov = get_normalized_images(I_fix, I_mov)

        patch_off = torch.tensor([
            [0, 0, 0], [0.02, 0, 0], [-0.02, 0, 0],
            [0, 0.02, 0], [0, -0.02, 0], [0, 0, 0.03], [0, 0, -0.03]
        ]).float()

        norm_roi_fn = build_norm_roi_fn(roi_pts, n_sample, pts_prob, W, H, D)

        results = {}

        # Config A: Macro-Only
        set_global_seed(42)
        m = ContinuousVPEF_Solver(hidden_dim=64)
        train_stage(m, norm_c, targets, edge_tensor, steps=170, lr=1.8e-3,
                    use_photometric=False)
        results["macro"] = (evaluate_tre(m, lm_n, sf, p0, p50, vs),) + evaluate_topology(m)

        # Config B: Micro-Only
        set_global_seed(42)
        m = ContinuousVPEF_Solver(hidden_dim=64)
        opt = torch.optim.Adam(m.parameters(), lr=1.5e-3)
        for _ in range(170):
            opt.zero_grad()
            norm_roi = norm_roi_fn()
            _, J = get_jacobian(m, norm_roi[:1024])
            disp_all = m(norm_roi)
            pts_f = (norm_roi.unsqueeze(1) + patch_off.unsqueeze(0)).view(-1, 3)
            pts_m = pts_f + disp_all.repeat_interleave(7, dim=0)
            v_f = F.grid_sample(T_fix, pts_f.view(1, 1, 1, -1, 3), align_corners=True).view(-1, 7)
            v_m = F.grid_sample(T_mov, pts_m.view(1, 1, 1, -1, 3), align_corners=True).view(-1, 7)
            fc = v_f - v_f.mean(dim=-1, keepdim=True)
            mc = v_m - v_m.mean(dim=-1, keepdim=True)
            lncc = (fc * mc).sum(dim=-1) / (torch.sqrt((fc ** 2).sum(dim=-1) * (mc ** 2).sum(dim=-1)) + 1e-4)
            loss = torch.mean(1.0 - lncc)
            det_F = torch.det(torch.eye(3).unsqueeze(0).expand(1024, -1, -1) + J)
            barrier = torch.where(det_F >= 1e-4, -torch.log(torch.clamp(det_F, min=1e-4)),
                                  -np.log(1e-4) + 1e4 * (1e-4 - det_F))
            loss = loss + 4.0 * torch.mean(torch.where(det_F < 0.05, barrier, torch.zeros_like(det_F)))
            loss.backward()
            opt.step()
        results["micro"] = (evaluate_tre(m, lm_n, sf, p0, p50, vs),) + evaluate_topology(m)

        # Config C: Single-Stage Coupled
        set_global_seed(42)
        m = ContinuousVPEF_Solver(hidden_dim=64)
        train_stage(m, norm_c, targets, edge_tensor, steps=170, lr=1.8e-3,
                    use_photometric=True, norm_roi_fn=norm_roi_fn,
                    T_fix=T_fix, T_mov=T_mov, patch_off=patch_off)
        results["coupled"] = (evaluate_tre(m, lm_n, sf, p0, p50, vs),) + evaluate_topology(m)

        # Config D: Full Two-Stage Decoupled (80 + 30 + 60)
        set_global_seed(42)
        m = ContinuousVPEF_Solver(hidden_dim=64)
        train_stage(m, norm_c, targets, edge_tensor, steps=80, lr=1.8e-3,
                    use_photometric=False)
        train_stage(m, norm_c, targets, edge_tensor, steps=30, lr=2.0e-4,
                    use_photometric=True, norm_roi_fn=norm_roi_fn,
                    T_fix=T_fix, T_mov=T_mov, patch_off=patch_off, anchor_w=0.3)
        train_stage(m, norm_c, targets, edge_tensor, steps=60, lr=1.3e-4,
                    use_photometric=True, norm_roi_fn=norm_roi_fn,
                    T_fix=T_fix, T_mov=T_mov, patch_off=patch_off, anchor_w=0.0)
        results["decoupled"] = (evaluate_tre(m, lm_n, sf, p0, p50, vs),) + evaluate_topology(m)

        print(f"  ✔ Case {cid:02d} ({time.time()-t_case:.0f}s) | "
              f"Macro {results['macro'][0]:.2f}mm/{results['macro'][1]:.2f}% | "
              f"Micro {results['micro'][0]:.2f}mm/{results['micro'][1]:.2f}% | "
              f"Coupled {results['coupled'][0]:.2f}mm/{results['coupled'][1]:.2f}% | "
              f"Decoupled {results['decoupled'][0]:.2f}mm/{results['decoupled'][1]:.2f}%")

        records.append({
            "Case": f"DIRLab-{cid:02d}",
            "Macro_TRE_mm": round(results["macro"][0], 3),
            "Macro_Fold_%": round(results["macro"][1], 4),
            "Micro_TRE_mm": round(results["micro"][0], 3),
            "Micro_Fold_%": round(results["micro"][1], 4),
            "Coupled_TRE_mm": round(results["coupled"][0], 3),
            "Coupled_Fold_%": round(results["coupled"][1], 4),
            "Decoupled_TRE_mm": round(results["decoupled"][0], 3),
            "Decoupled_Fold_%": round(results["decoupled"][1], 4),
        })

    df = pd.DataFrame(records)
    out_csv = os.path.join(OUT_DIR, "Table_Validation_Scale_Decoupling.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  -> Aligned ablation saved: {out_csv}")
    print(f"  Total ablation time: {time.time()-t_all:.0f}s\n")
    return df


# =====================================================================
# 3. Robustness: anchor density (relative ratio) & noise
# =====================================================================
def run_anchor_density_benchmark(cases=None, n_seeds=1):
    if cases is None:
        cases = [("DIRLAB", i) for i in range(1, 6)] + [("POPI", i) for i in range(1, 4)]

    print("\n" + "=" * 88)
    print(f">>> [TASK 3/4] ANCHOR DENSITY & NOISE ROBUSTNESS ({len(cases)} cases, seeds={n_seeds})")
    print("    Sampling ratios: 20% / 40% / 60% / 80% / 100% of available anchors")
    print("=" * 88)

    ratios = [0.20, 0.40, 0.60, 0.80, 1.00]
    noise_sigmas = [0.0, 0.5, 1.0, 1.5]
    all_records = []
    t_all = time.time()

    for ds, cid in cases:
        try:
            I_fix, I_mov, p0, p50, vs, _, _ = load_case_data(ds, cid)
            H, W, D = I_fix.shape
            is_popi = (ds == "POPI")
            sf = torch.tensor([(W - 1) / 2.0, (H - 1) / 2.0, (D - 1) / 2.0]).float()
            lm_n = torch.from_numpy(np.stack([
                (p0[:, 0] / (W - 1)) * 2 - 1,
                (p0[:, 1] / (H - 1)) * 2 - 1,
                (p0[:, 2] / (D - 1)) * 2 - 1
            ], axis=-1)).float()

            anchors = extract_vessel_anchors_unified(I_fix, cid, is_popi=is_popi)
            disps_vox, valid = dual_track_matcher(I_fix, I_mov, anchors, cid, is_popi=is_popi, vs=vs)
            anc_v, disp_v = anchors[valid], disps_vox[valid]
            n_avail = len(anc_v)

            for ratio in ratios:
                n_sel = max(15, int(n_avail * ratio))
                for noise in noise_sigmas:
                    for seed_idx in range(n_seeds):
                        seed_val = 42 + seed_idx * 1000
                        set_global_seed(seed_val)
                        sel = np.random.choice(n_avail, n_sel, replace=False)
                        sub_anc = anc_v[sel]
                        sub_disp = disp_v[sel].copy()
                        if noise > 0.0:
                            sub_disp += np.random.normal(0, noise, sub_disp.shape).astype(np.float32)

                        norm_c = torch.from_numpy(np.stack([
                            (sub_anc[:, 0] / (W - 1)) * 2 - 1,
                            (sub_anc[:, 1] / (H - 1)) * 2 - 1,
                            (sub_anc[:, 2] / (D - 1)) * 2 - 1
                        ], axis=-1)).float()
                        targets = torch.from_numpy(sub_disp).float() / sf

                        model = ContinuousVPEF_Solver(hidden_dim=64)
                        opt = torch.optim.Adam(model.parameters(), lr=1.8e-3)
                        for _ in range(80):
                            opt.zero_grad()
                            loss_d = torch.mean((model(norm_c) - targets) ** 2)
                            pde_pts = torch.rand(1024, 3) * 2 - 1
                            _, J = get_jacobian(model, pde_pts)
                            det_F = torch.det(torch.eye(3).unsqueeze(0).expand(1024, -1, -1) + J)
                            barrier = torch.where(
                                det_F >= 1e-4,
                                -torch.log(torch.clamp(det_F, min=1e-4)),
                                -np.log(1e-4) + 10000.0 * (1e-4 - det_F)
                            )
                            l_topo = 4.0 * torch.mean(torch.where(det_F < 0.05, barrier, torch.zeros_like(det_F)))
                            (loss_d + 0.002 * torch.sum(J ** 2, dim=[-1, -2]).mean() + l_topo).backward()
                            opt.step()

                        tre = evaluate_tre(model, lm_n, sf, p0, p50, vs)
                        all_records.append({
                            "Dataset": ds,
                            "Case": f"{ds}-{cid:02d}",
                            "Anchor_Ratio_%": int(ratio * 100),
                            "N_Available": n_avail,
                            "N_Selected": n_sel,
                            "Noise_Sigma_vox": noise,
                            "Seed": seed_val,
                            "TRE_mm": round(tre, 3),
                        })
            print(f"  ✔ [{ds}-{cid:02d}] done (N_available={n_avail})")
        except Exception as e:
            print(f"  ⚠ [{ds}-{cid:02d}] skipped: {e}")

    df = pd.DataFrame(all_records)
    if len(df) > 0:
        agg = df.groupby(["Anchor_Ratio_%", "Noise_Sigma_vox"])["TRE_mm"].agg(
            ["mean", "std", "count"]).reset_index()
        agg.columns = ["Anchor_Ratio_%", "Noise_Sigma_vox", "TRE_Mean_mm", "TRE_STD_mm", "N_Cases"]
        print("\n>>> Aggregated anchor-ratio convergence (mean over cases & seeds):")
        print(agg.to_string(index=False))
        agg.to_csv(os.path.join(OUT_DIR, "Table_Validation_Anchor_Aggregated.csv"), index=False)

    out_csv = os.path.join(OUT_DIR, "Table_Validation_Anchor_Robustness.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  -> Anchor robustness table saved: {out_csv}")
    print(f"  Total robustness time: {time.time()-t_all:.0f}s\n")
    return df


# =====================================================================
# 4. Stats: Holm-Bonferroni correction & effect sizes
# =====================================================================
def run_statistical_correction_audit():
    print("\n" + "=" * 88)
    print(">>> [TASK 4/4] STATISTICAL CORRECTION & EFFECT-SIZE AUDIT")
    print("=" * 88)

    f_t6 = os.path.join(RESULT_ROOT, "Table_6_AAPM_TG132_Definitive.csv")
    if not os.path.exists(f_t6):
        print(f"  ⚠ Table_6 not found at {f_t6}. Run benchmark_4D.py first.")
        return None

    df = pd.read_csv(f_t6)

    def cohens_dz(x, y):
        diff = np.asarray(x) - np.asarray(y)
        return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))

    def bootstrap_ci(x, y, n_boot=5000, seed=42):
        x, y = np.asarray(x), np.asarray(y)
        rng = np.random.RandomState(seed)
        diffs = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(x), len(x))
            diffs.append(np.mean(x[idx] - y[idx]))
        return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    endpoints = [
        ("T50 Gamma 3mm/3%", "Gamma_T50_4D_%", "Gamma_T50_Chained_%", "higher"),
        ("Target Residual TRE", "Target_TRE_Chained_mm", "Target_TRE_4D_mm", "lower"),
        ("Normal Lung V40Gy", "V40_Leak_Chained_cc", "V40_Leak_4D_cc", "lower"),
        ("AAPM TG-101 R50%", "R50_Chained", "R50_4D", "lower"),
        ("Whole-Lung MLD", "MLD_Total_Chained_Gy", "MLD_Total_4D_Gy", "lower"),
        ("Normal Lung V20Gy", "V20_Chained_%", "V20_4D_%", "lower"),
    ]

    p_vals, rows = [], []
    for label, col_a, col_b, _ in endpoints:
        if col_a in df.columns and col_b in df.columns:
            val_a, val_b = df[col_a].values, df[col_b].values
            if np.all(np.abs(val_a - val_b) < 1e-12):
                continue
            stat, p = wilcoxon(val_a, val_b, alternative='greater')
            dz = cohens_dz(val_a, val_b)
            ci_lo, ci_hi = bootstrap_ci(val_a, val_b)
            p_vals.append(p)
            rows.append({
                "Endpoint": label,
                "Raw_p_value": p,
                "Cohens_dz": round(dz, 3),
                "95%_CI_lower": round(ci_lo, 4),
                "95%_CI_upper": round(ci_hi, 4),
            })

    order = np.argsort(p_vals)
    adj_p = np.zeros(len(p_vals))
    prev = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, p_vals[idx] * (len(p_vals) - rank))
        val = max(val, prev)
        adj_p[idx] = val
        prev = val

    for i in range(len(rows)):
        rows[i]["Holm_Adjusted_p"] = round(adj_p[i], 4)
        rows[i]["Significant_at_0.05"] = "Yes" if adj_p[i] < 0.05 else "No"

    df_res = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "Table_Validation_Statistical_Correction.csv")
    df_res.to_csv(out_csv, index=False)
    print(df_res.to_string(index=False))
    print(f"\n  -> Statistical correction table saved: {out_csv}\n")
    return df_res


# =====================================================================
# Main dispatcher
# =====================================================================
def _already_exists(name):
    return os.path.exists(os.path.join(OUT_DIR, name))


def main():
    parser = argparse.ArgumentParser(description="SD-DNF Benchmark Validation Suite v3")
    parser.add_argument("--task", type=str,
                        choices=["baseline", "ablation", "robustness", "stats", "all"],
                        default="all")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of random seeds for anchor robustness (default: 1)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run baseline/stats even if output exists")
    args = parser.parse_args()

    print(f"\n[Paths]")
    print(f"  Script     : {os.path.abspath(__file__)}")
    print(f"  Result root: {RESULT_ROOT}")
    print(f"  Output dir : {OUT_DIR}")

    t_start = time.time()

    # Baseline
    if args.task in ["baseline", "all"]:
        if args.task == "all" and _already_exists("Table_Validation_Baseline.csv") and not args.force:
            print("\n>>> [TASK 1/4] BASELINE skipped (output exists; use --force to override)")
        else:
            run_baseline_study(cases=[1, 2, 3, 4, 5])

    # Ablation
    if args.task in ["ablation", "all"]:
        run_scale_decoupling_ablation(cases=[1, 2, 3, 4, 5])

    # Robustness
    if args.task in ["robustness", "all"]:
        run_anchor_density_benchmark(n_seeds=args.seeds)

    # Stats
    if args.task in ["stats", "all"]:
        if args.task == "all" and _already_exists("Table_Validation_Statistical_Correction.csv") and not args.force:
            print("\n>>> [TASK 4/4] STATS skipped (output exists; use --force to override)")
        else:
            run_statistical_correction_audit()

    print("=" * 88)
    print(f"  All requested tasks completed. Total time: {time.time() - t_start:.1f} s")
    print(f"  Artifacts archived at: {OUT_DIR}")
    print("=" * 88)


if __name__ == "__main__":
    main()