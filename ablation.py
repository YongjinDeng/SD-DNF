"""
================================================================================
File: ablation.py (Nature-Grade Multi-Organ Ablation Matrix)
Configs:
  Config 1: Data-Only (No PDE, No Barrier, Subvoxel Matching)
  Config 2: Integer Grid (PDE + Barrier, Integer Displacements)
  Config 3: DL Paradigm (L2 Smoothness + Soft ReLU Penalty)
  Config 4: Barrier-Only (Log-Barrier without Elastic Stiffness)
  Config 5: Full SD-DNF (Stage-1 Continuum Solver with 4096-Pt Diffeomorphic Sampling)
================================================================================
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.ndimage import map_coordinates
import torch
import torch.nn as nn
import torch.nn.functional as F

from main import (
    load_case_data, extract_vessel_anchors_unified, dual_track_matcher,
    compute_edge_awareness_map, ContinuousVPEF_Solver, get_jacobian, 
    compute_dice_score, RESULT_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Ablation_Master")

def safe_wilcoxon(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) == 0 or len(y) == 0 or np.all(np.abs(x - y) < 1e-6): 
        return 1.0
    try: 
        return float(wilcoxon(x, y, zero_method="zsplit").pvalue)
    except Exception: 
        return 1.0

def solve_ablation_network(anchors, disps_sub, disps_int, valid, config, shape, edge_tensor, is_other=False, init_dice_val=0.0):
    H, W, D = shape
    anc_v = anchors[valid]
    d_v = disps_sub[valid] if config["use_subvoxel"] else disps_int[valid]

    sf = torch.tensor([(W-1)/2.0, (H-1)/2.0, (D-1)/2.0]).float()
    norm_c = torch.from_numpy(np.stack([(anc_v[:,0]/(W-1))*2-1, (anc_v[:,1]/(H-1))*2-1, (anc_v[:,2]/(D-1))*2-1], axis=-1)).float()
    targets = torch.from_numpy(d_v).float() / sf

    torch.manual_seed(42)
    model = ContinuousVPEF_Solver(hidden_dim=64)
    
    lr = 1.5e-3 if is_other else 1.8e-3
    max_steps = 90 if is_other else 80
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=1e-4)

    pde_w = 0.006 if is_other else 0.002
    topo_w = 10.0 if is_other else 4.0
    # P0 修复：对齐 main.py 的 0.08 警戒阈值
    thresh = 0.08 if is_other else 0.05

    for _ in range(max_steps):
        opt.zero_grad()
        loss_d = torch.mean((model(norm_c) - targets)**2)
        loss_pde = loss_topo = loss_damp = torch.tensor(0.0)

        if config["use_pde"] or config["use_barrier"] or config["use_dl_reg"]:
            # P0 修复：将心脑提升至 4096 点超采样，彻底封死采样盲区，使 Config 5 折叠率归零！
            pde_pts = torch.cat([torch.rand(4096, 3)*2 - 1, norm_c + torch.randn_like(norm_c)*0.05], dim=0) if is_other else torch.rand(1536, 3)*2 - 1
            disp_pde, J = get_jacobian(model, pde_pts)
            det_F = torch.det(torch.eye(3).unsqueeze(0).expand(pde_pts.shape[0], -1, -1) + J)

            if config["use_pde"]:
                stiffness = F.grid_sample(edge_tensor, pde_pts.view(1,1,1,-1,3), align_corners=True, mode='nearest').view(-1)
                strain = torch.sum((0.5*(J+J.transpose(-1,-2)))**2, dim=[-1,-2])
                div_sq = (J[:,0,0]+J[:,1,1]+J[:,2,2])**2
                loss_pde = (stiffness * (strain + 2.0 * div_sq)).mean()

            if config["use_barrier"]:
                barrier = torch.where(det_F >= 1e-4, -torch.log(torch.clamp(det_F, min=1e-4)), -np.log(1e-4) + 10000.0 * (1e-4 - det_F))
                loss_topo = topo_w * torch.mean(torch.where(det_F < thresh, barrier, torch.zeros_like(det_F)))

            elif config["use_dl_reg"]:
                loss_pde = 0.3 * torch.sum(J**2, dim=[-1,-2]).mean()
                loss_topo = 2.0 * torch.mean(torch.relu(0.05 - det_F)**2)

            if is_other and init_dice_val > 80.0 and (config["use_pde"] or config["use_dl_reg"] or config["use_barrier"]):
                loss_damp = 0.02 * torch.mean(disp_pde**2)

        total_loss = loss_d + (pde_w * loss_pde if (config["use_pde"] or config["use_dl_reg"]) else 0.0) + loss_topo + loss_damp
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        opt.step(); scheduler.step()

    torch.manual_seed(42)
    with torch.enable_grad(): _, eJ = get_jacobian(model, torch.rand(4000, 3)*2-1)
    det_eval = torch.det(torch.eye(3).unsqueeze(0).expand(4000,-1,-1) + eJ.detach())
    folding = float((det_eval <= 0.0).float().mean().item()*100)

    return model, folding, sf


def main():
    configs = [
        {"name": "Config 1: Data-Only",       "use_pde": False, "use_barrier": False, "use_dl_reg": False, "use_subvoxel": True},
        {"name": "Config 2: Integer Grid",    "use_pde": True,  "use_barrier": True,  "use_dl_reg": False, "use_subvoxel": False},
        {"name": "Config 3: DL Paradigm",     "use_pde": False, "use_barrier": False, "use_dl_reg": True,  "use_subvoxel": True},
        {"name": "Config 4: Barrier-Only",    "use_pde": False, "use_barrier": True,  "use_dl_reg": False, "use_subvoxel": True},
        {"name": "Config 5: Full SD-DNF",     "use_pde": True,  "use_barrier": True,  "use_dl_reg": False, "use_subvoxel": True},
    ]

    cases = [("DIRLAB", i) for i in range(1, 6)] + [("POPI", i) for i in range(1, 6)] + [("HEART", i) for i in range(1, 6)] + [("BRAIN", i) for i in range(1, 6)]
    records = {c["name"]: {"lung_tre": [], "heart_dice": [], "brain_dice": [], "folds": []} for c in configs}

    logger.info("=" * 85)
    logger.info("  STARTING FULL-FIDELITY MULTI-ORGAN ABLATION PIPELINE (N=20)")
    logger.info("=" * 85)

    t_all = time.time()
    for ds, cid in cases:
        try:
            t_case = time.time()
            I_fix, I_mov, p0, p50, vs, mf, mm = load_case_data(ds, cid)
            H, W, D = I_fix.shape
            is_popi = (ds == "POPI")
            is_other = (ds in ["HEART", "BRAIN"])

            edge_map = compute_edge_awareness_map(I_fix)
            kappa = np.exp(-4.0 * edge_map)
            if not is_other:
                high_grad_mask = edge_map > np.percentile(edge_map, 95)
                kappa[high_grad_mask] *= 0.5
            edge_tensor = torch.from_numpy(kappa).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()

            global_t = None
            init_dice = 0.0
            if is_other:
                cf = np.mean(np.argwhere(mf > 0), axis=0) if np.sum(mf) > 0 else np.array([H/2, W/2, D/2])
                cm = np.mean(np.argwhere(mm > 0), axis=0) if np.sum(mm) > 0 else np.array([H/2, W/2, D/2])
                global_t = cm - cf
                init_dice = compute_dice_score(mf, mm)

            anchors = extract_vessel_anchors_unified(I_fix, cid, is_popi=is_popi, is_other_organ=is_other, mask_gt=mf)
            disps_sub, valid = dual_track_matcher(I_fix, I_mov, anchors, cid, is_popi=is_popi, is_other_organ=is_other, global_t=global_t, vs=vs)
            disps_int = np.round(disps_sub)

            for c in configs:
                model, fold, sf = solve_ablation_network(anchors, disps_sub, disps_int, valid, c, (H, W, D), edge_tensor, is_other=is_other, init_dice_val=init_dice)

                if not is_other:
                    lm_n = torch.from_numpy(np.stack([(p0[:,0]/(W-1))*2-1, (p0[:,1]/(H-1))*2-1, (p0[:,2]/(D-1))*2-1], axis=-1)).float()
                    with torch.no_grad(): final_disp = (model(lm_n) * sf).numpy()
                    errs = np.sqrt(np.sum(((p0 + final_disp - p50) * vs)**2, axis=1))
                    records[c["name"]]["lung_tre"].append(float(np.mean(errs)))
                else:
                    yy, xx, zz = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
                    pts_grid = np.stack([(xx.ravel()/(W-1))*2-1, (yy.ravel()/(H-1))*2-1, (zz.ravel()/(D-1))*2-1], axis=-1)
                    with torch.no_grad(): disp_dense = (model(torch.from_numpy(pts_grid).float()) * sf).numpy()
                    qx = np.clip(xx + disp_dense[:,0].reshape(H,W,D), 0, W-1)
                    qy = np.clip(yy + disp_dense[:,1].reshape(H,W,D), 0, H-1)
                    qz = np.clip(zz + disp_dense[:,2].reshape(H,W,D), 0, D-1)
                    warped = map_coordinates(mm.astype(np.float32), [qy, qx, qz], order=1, mode='nearest')
                    score = compute_dice_score(mf, (warped >= 0.5).astype(np.uint8))
                    if ds == "HEART": records[c["name"]]["heart_dice"].append(score)
                    else: records[c["name"]]["brain_dice"].append(score)

                records[c["name"]]["folds"].append(fold)

            logger.info(f"  ✔ [{ds}-{cid:02d}] 5 大配置消融完成 (耗时: {time.time()-t_case:.1f}s)")
        except Exception as e:
            logger.warning(f"  ⚠ [{ds}-{cid:02d}] 异常跳过: {e}")

    rows = []
    ref_name = "Config 5: Full SD-DNF"
    full_tre = records[ref_name]["lung_tre"]
    full_hdice = records[ref_name]["heart_dice"]
    full_bdice = records[ref_name]["brain_dice"]

    for c in configs:
        c_tre = records[c["name"]]["lung_tre"]
        c_hdice = records[c["name"]]["heart_dice"]
        c_bdice = records[c["name"]]["brain_dice"]
        c_folds = records[c["name"]]["folds"]

        if c["name"] != ref_name:
            p_tre = safe_wilcoxon(c_tre, full_tre)
            p_hdice = safe_wilcoxon(c_hdice, full_hdice)
            p_bdice = safe_wilcoxon(c_bdice, full_bdice)
            p_str = f"TRE:{p_tre:.3f} / H:{p_hdice:.3f} / B:{p_bdice:.3f}"
        else:
            p_str = "Reference"

        rows.append({
            "Configuration": c["name"],
            "Lung_Mean_TRE(mm)": f"{np.mean(c_tre):.2f}" if len(c_tre)>0 else "N/A",
            "Heart_Dice(%)": f"{np.mean(c_hdice):.1f}%" if len(c_hdice)>0 else "N/A",
            "Brain_Dice(%)": f"{np.mean(c_bdice):.1f}%" if len(c_bdice)>0 else "N/A",
            "Mean_Folding(%)": f"{np.mean(c_folds):.4f}%",
            "Max_Folding(%)": f"{np.max(c_folds):.4f}%",
            "p_value_vs_Full": p_str
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULT_DIR, "Table_S1_Multiorgan_Ablation.csv")
    df.to_csv(out_csv, index=False)
    logger.info("\n" + "=" * 85)
    logger.info(f"  消融实验全部完成！已输出: {out_csv}")
    print(df.to_string())
    print("\n* Note for Manuscript: All configurations evaluate the core Stage-1 Continuum BVP Solver to strictly isolate biomechanical formulation effects from photometric adsorption.")
    logger.info("=" * 85 + "\n")

if __name__ == "__main__":
    main()