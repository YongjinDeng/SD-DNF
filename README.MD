# SD-DNF: Scale-Decoupled Diffeomorphic Neural Fields

Official PyTorch implementation of the paper:  
**"Scale-Decoupled Diffeomorphic Neural Fields for Topology-Certified Registration and 4D Radiotherapy Dose Verification"**  
*(Under review at IEEE Transactions on Medical Imaging)*

---

## ⚡ Highlights
- **Data-Free & Instance-Specific:** Solves non-rigid registration as a continuum boundary-value problem without requiring offline population pre-training.
- **Topology-Certified:** Guarantees strict local invertibility ($\det(\mathbf{F}) > 0$) via an analytical logarithmic interior-point barrier, achieving 0.0000% empirical folding.
- **4D Lagrangian Dosimetry:** Continuous spatiotemporal trajectory integration eliminating dead-reckoning drift in dynamic SBRT delivery conforming to AAPM TG-132.
- **CPU Sustainable:** Pure CPU execution within 15–50 s per standard volume (<70 J active energy).

---

## 🛠️ Environment & Requirements

Recommended: Python 3.8+ and PyTorch (CPU or GPU).

Install standard dependencies:
```bash
pip install torch numpy scipy pandas pydicom psutil matplotlib seaborn

🚀 Quick Start

1. 3D Multi-Organ Benchmark

To run the primary 3D registration pipeline on thoracic CT and MRI cohorts:

python main.py

2. 4D Spatiotemporal Tracking & Dosimetric Audit

To evaluate 20-step continuous Lagrangian dose accumulation against AAPM TG-132
/ QUANTEC protocols:

python benchmark_4D.py

3. Ablation Study & Supplementary Benchmarks

# Symmetrical multi-organ ablation matrix (Table III)
python ablation.py

# Synthetic vortex stress test and parameter sensitivity (Fig. S1, Table S-III)
python benchmark_supplementary.py

# Multi-seed reproducibility study (Table S-IV)
python benchmark_multiseed_idir.py

📂 Public Datasets

  - DIR-Lab 4DCT: https://www.dir-lab.com
  - POPI 4DCT: CREATIS Open Dataset
  - Medical Segmentation Decathlon (MSD): Task02 (Heart) and Task04
    (Hippocampus)

📖 Citation

If you find this work or code useful for your research, please cite:

@article{deng2026sddnf,
  title={Scale-Decoupled Diffeomorphic Neural Fields for Topology-Certified Registration and 4D Radiotherapy Dose Verification},
  author={Deng, Yongjin and Wang, Chengtao and Pan, Junxi and Wu, Shuang and Bao, Yong},
  journal={Under Review at IEEE Transactions on Medical Imaging},
  year={2026}
}

📄 License

This project is open-source and released under the MIT License.

