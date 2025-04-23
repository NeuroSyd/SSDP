**SSDP** (Spike‑Synchrony‑Dependent Plasticity) is a biologically inspired learning rule that updates synaptic weights according to *group‑level* spike synchrony rather than isolated spike pairs.  
This repository provides clean, task‑ready implementations of SSDP across three representative spiking‑neural‑network architectures:


| Folder | Integration target | Brief description |
|--------|--------------------|-------------------|
| `SSD‑ViT/`             | **SSD-ViT**  | Plug‑and‑play SSDP module added to the classifier and last‑stage DSSA projection weights |
| `Single_layer_SSDP/`   | Custom single‑hidden‑layer SNN         | Minimal example for quickly understanding and testing SSDP |
| `SSDP_DH_SNN/`         | **DH‑SNN** (Dendritic & Heterogeneous SNN) | Demonstrates SSDP inside the publicly available DH‑SNN codebase |
| `Visualization/`       | Analysis utilities                     | Scripts for t‑SNE, PCA, spike‑rate plots, etc. |

Quick Start


2.1 SSD‑ViT

cd SSD-ViT
python train.py --config cifar100.yaml


Dataset & config follow the original SpikingResformer repo. link: https://github.com/xyshi2000/SpikingResformer
Only change: run `train.py` instead of `main.py`.


Where SSDP is inserted: classifier weights + last DSSA Wproj.
Hyper‑parameters (A_plus, sigma, etc.) match those reported in the paper’s Supplementary Information.

2.2 Single‑layer SNN 

cd Single_layer_SSDP
python fashion_MNIST.py

2.3 SSDP with DH-SNN

Dataset & config follow the original DH-SNN repo. link: https://github.com/eva1801/DH-SNN

cd SSDP_DH_SNN
python train_ssdp.py

License
This project is released under the MIT License—see LICENSE for details.

Enjoy exploring synchrony‑driven learning! If you run into issues or have feature requests, feel free to email: ytia0587@uni.sydney.edu.au.






