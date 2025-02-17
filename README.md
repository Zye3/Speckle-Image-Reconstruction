# Seeing Through Multimode Fibers Using a Real-Valued Intensity Transmission Matrix with Deep Learning
**Ziyu Ye, Tianrui Zhao, Wenfeng Xia\***

## Overview
Our approach first characterizes the multimode fiber (**MMF**) to retrieve images using a **Real-Valued Intensity Transmission Matrix (RVITM)** algorithm. We then refine the reconstructions with a **Hierarchical, Parallel Multi-Scale (HPM) Attention U-Net** to further improve image quality. Experimental results demonstrate that our approach achieves high-quality reconstructions, with **Structural Similarity Index (SSIM)** and **Peak Signal-to-Noise Ratio (PSNR)** values of up to **0.9524** and **33.244 dB**, respectively.

---

## System Requirements

### Operating System
All networks have been tested on **Linux (Ubuntu 20.04)**, **Windows**, and **macOS**. They should work out of the box.

### Hardware Requirements
- **Recommended Device:** GPU  
- **Supported Devices:** GPU, CPU, and Apple M1/M2  
- **Training:** A GPU with at least **10 GB** of VRAM is recommended (e.g., **RTX 2080 Ti**, **RTX 3080/3090**, or **RTX 4080/4090**). Training on CPU or M1/M2 (**MPS**) will be significantly slower.  
- **CPU Requirements:** At least 6 cores (12 threads) are recommended, especially for data augmentation when training. The faster the GPU, the more powerful the CPU should be.

---

## Installation Instructions

1. **Install PyTorch**  
   Follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) (via conda/pip) and choose the version that supports your hardware (**CUDA**, **MPS**, or **CPU**).  
   *For maximum speed, advanced users may consider [compiling PyTorch from source](https://github.com/pytorch/pytorch#from-source).*

2. **Environment Setup**  
   You can create a Conda environment using the provided [YAML file](https://github.com/Zye3/Speckle-Image-Reconstruction/blob/master/pytorch.yaml), which is based on **Python 3.9**:
   ```bash
   conda env create -f pytorch.yaml
   conda activate pytorch


>Note:You can use this __[yaml file](https://github.com/Zye3/Speckle-Image-Reconstruction/blob/master/pytorch.yaml)__ based on Python 3.9

## Dataset
You can download the dataset using this __[Google Drive link.](https://drive.google.com/drive/folders/1X6b-sYlbe0j2Sp6p-3X7M7HJE2f4iPwn)__


## Real-Vauled Intensity Transmission Matrix for image retrieval __[Matlab Code Here](https://github.com/Zye3/Speckle-Image-Reconstruction/blob/master/calculate_RVITM.m)__
We use __RVITM__ to retrieve images through the MMF.

| Model         | SSIM    | PSNR    |
|---------------|---------|---------|
| RVITM         | 0.1665  | 9.8718  |

## Deep learning for image reconstruction
Below are several deep learning models to choose from:

| Model         | SSIM    | PSNR    |
|---------------|---------|---------|
| FC-AE         | 0.7531  | 12.3864 |
| CNN-AE        | 0.8550  | 25.5408 |
| AE-SNN        | 0.9364  | 16.0765 |
| U-Net         | 0.9368  | 32.7797 |
| Att-U-Net     | 0.9404  | 32.2254 |
| HPM-Att-U-Net(__Ours__) | __0.9524__  | __33.2440__ |

## Citation

Please cite the following paper when using this code:
1. Z. Ju, Z. Yu, Z. Meng, N. Zhan, L. Gui, K. Xu, Simultaneous illumination and imaging based on a single multimode fibre, Opt. Express 30 (9) (2022) 15596.
2. Y. Li, Z. Yu, Y. Chen, T. He, J. Zhang, R. Zhao, K. Xu, Image Reconstruction Using Pre-Trained Autoencoder on Multimode Fiber Imaging System, IEEE Photonics Technol. Lett. 32 (13) (2020) 779–782.
3. H. Chen, Z. He, Z. Zhang, Y. Geng, W. Yu, Binary amplitude-only image reconstruction through an MMF based on an AE-SNN combined deep learning model, Opt. Express 28 (20) (2020) 30048–30062.
4. O. Ozan, J. Schlemper, L., L. Folgoc, M. Lee, M. Heinrich, K. Misawa, K. Mori et al. "Attention U-Net: Learning Where to Look for the Pancreas." CVPR, (2018).
5. O. Ronneberge, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image.





