# [Seeing Through Multimode Fibers Using a Real-Valued Intensity Transmission Matrix with Deep Learning](https://opg.optica.org/oe/fulltext.cfm?uri=oe-33-7-16222&id=569946)
**Ziyu Ye, Tianrui Zhao, Wenfeng Xia\***

## Overview
Our approach first characterizes the multimode fiber (**MMF**) to retrieve images using a **Real-Valued Intensity Transmission Matrix (RVITM)** algorithm. We then refine the reconstructions with a **Hierarchical, Parallel Multi-Scale (HPM) Attention U-Net** to further improve image quality. Experimental results demonstrate that our approach achieves high-quality reconstructions, with **Structural Similarity Index (SSIM)** and **Peak Signal-to-Noise Ratio (PSNR)** values of up to **0.9524** and **33.244 dB**, respectively.

<img width="375" alt="image" src="https://github.com/user-attachments/assets/7f54798d-7132-4eb4-8f50-7b47590e0ae5" />


## System Requirements

### Operating System
All networks have been tested on **Linux (Ubuntu 20.04)**, **Windows**, and **macOS**. They should work out of the box.

### Hardware Requirements
- **Recommended Device:** GPU  
- **Supported Devices:** GPU, CPU, and Apple M1/M2  
- **Training:** A GPU with at least **10 GB** of VRAM is recommended (e.g., **RTX 2080 Ti**, **RTX 3080/3090**, or **RTX 4080/4090**). Training on CPU or M1/M2 (**MPS**) will be significantly slower.  
- **CPU Requirements:** At least 6 cores (12 threads) are recommended, especially for data augmentation when training. The faster the GPU, the more powerful the CPU should be.

## Installation Instructions

1. **Install PyTorch**  
   Follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) (via conda/pip) and choose the version that supports your hardware (**CUDA**, **MPS**, or **CPU**).  
   *For maximum speed, advanced users may consider [compiling PyTorch from source](https://github.com/pytorch/pytorch#from-source).*

2. **Python packages**

   `pytorch.yaml` is a Windows Anaconda export. On Linux, install the packages with:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The public download is this [Google Drive folder](https://drive.google.com/drive/folders/1avbOZG4P4LPlVDdpRAyIuZBGf_8W8-gv). Opening it asks for a Google sign-in.

The training code does not read those PNGs itself. `data_preprocess.py` still loads three NumPy arrays:

| Argument | Array | What the loader does |
|---|---|---|
| `--data_dir` 1st path | input, shape `(N, H, W)` | divides by `--normalize_factor` (default 256) |
| `--data_dir` 2nd path | label, shape `(N, H, W)` | cast to float and kept as stored |
| `--data_dir` 3rd path | index, integer ids | selects rows of the two arrays |

Pair the files by the same integer id before you build those arrays:

| File | Meaning |
|---|---|
| `captured_part_*/<size>_<size>/Data_<id>.png` | RVITM recovery. 16-bit grey, values span 0–65535 |
| `part_*/Label_<id>.png` | ground truth shown on the DMD. 8-bit grey, values span 0–255 |

`Image0_*.tif` under `2.captured_reference` is the calibration speckle, not the network input. Folders named `dark_strategy_*` are not the pair used here.

The network ends in a sigmoid, and `test.py` scores SSIM / PSNR with `data_range=1`. Store both arrays in `[0, 1]` after the loader runs. With the loader left unchanged, that means:

```python
input_npy[i] = (Data_<id> / 65535) * 256   # loader divides by 256
label_npy[i] = Label_<id> / 255            # loader keeps this value
index_npy = np.arange(N)
```

A 64×64 tree that matches this layout is `1.object_mnist_64` (also Fashion-MNIST and CIFAR at 64). A 32×32 tree is `1.object_mnist_32` and the other `1.object_*_32` folders. `--model_type AE` and `AE-SNN` are fully connected and only accept 64×64. `CNN-AE`, `U-net`, `Att_UNet`, and `HPM_UNet` accept 32 and 64.

```bash
python main.py --data_dir input.npy label.npy index.npy \
  --model_type HPM_UNet --code_length 64 --epochs 1 --batch_size 4

python test.py --data_dir input.npy label.npy index.npy \
  --model_type HPM_UNet --ckpt runs/<run>/HPM_UNet.pth
```

`--code_length` is only the spatial size used when exporting ONNX. It does not resize the images or rebuild `AE` / `AE-SNN`.

## Real-Vauled Intensity Transmission Matrix for image retrieval 
We use __RVITM__ to retrieve images through the MMF.
__[Matlab Code Here](https://github.com/Zye3/Speckle-Image-Reconstruction/blob/master/calculate_RVITM.m)__

| Model         | SSIM    | PSNR    |
|---------------|---------|---------|
| RVITM         | 0.1665  | 9.8718  |

## Deep learning for image reconstruction
Below are several deep learning models to choose from __[model.py](https://github.com/Zye3/Speckle-Image-Reconstruction/blob/master/model.py)__:

| Model         | SSIM    | PSNR    |
|---------------|---------|---------|
| FC-AE         | 0.7531  | 12.3864 |
| CNN-AE        | 0.8550  | 25.5408 |
| AE-SNN        | 0.9364  | 16.0765 |
| U-Net         | 0.9368  | 32.7797 |
| Att-U-Net     | 0.9404  | 32.2254 |
| HPM-Att-U-Net(__Ours__) | __0.9524__  | __33.2440__ |

Command-line names: FC-AE is `--model_type AE`, U-Net is `U-net`, Att-U-Net is `Att_UNet`, HPM-Att-U-Net is `HPM_UNet`. `AE` and `AE-SNN` take 64×64 inputs.

## Citation

Please cite the following paper when using this code:

1.T. Zhao, S. Ourselin, T. Vercauteren and W. Xia. “Seeing through multimode fibers with real-valued intensity transmission matrices,” Optics Express 28, 20978 – 20991 (2020).

2.__[Z. Ye, T. Zhao, and W. Xia, "Seeing through multimode fibers using real-valued intensity transmission matrix with deep learning," Opt. Express 33, 16222-16236 (2025)](https://doi.org/10.1364/OE.553949)__






