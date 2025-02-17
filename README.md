# Seeing through multimode fibres using a real-valued intensity transmission matrix with deep learning
Ziyu Ye,Tianrui Zhao,Wenfeng Xia*

Our approach first characterises the MMF and retrieves images using a Real-Vauled Intensity Transmission Matrix (RVITM) algorithm, followed by refinement with a Hierarchical, Parallel Multi-Scale (HPM)-Attention U-Net to improve image quality. Experimental results demonstrated that our approach achieved high-quality reconstructions, with Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR) values of up to 0.9524 and 33.244 dB, respectively.
Please cite the following paper when using this code:
Z. Ju, Z. Yu, Z. Meng, N. Zhan, L. Gui, K. Xu, Simultaneous illumination and imaging based on a single multimode fibre, Opt. Express 30 (9) (2022) 15596.
Y. Li, Z. Yu, Y. Chen, T. He, J. Zhang, R. Zhao, K. Xu, Image Reconstruction Using Pre-Trained Autoencoder on Multimode Fiber Imaging System, IEEE Photonics Technol. Lett. 32 (13) (2020) 779–782.
H. Chen, Z. He, Z. Zhang, Y. Geng, W. Yu, Binary amplitude-only image reconstruction through an MMF based on an AE-SNN combined deep learning model, Opt. Express 28 (20) (2020) 30048–30062.
O. Ozan, J. Schlemper, L., L. Folgoc, M. Lee, M. Heinrich, K. Misawa, K. Mori et al. "Attention U-Net: Learning Where to Look for the Pancreas." CVPR, (2018).
O. Ronneberge, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image.

1. Real-Vauled Intensity Transmission Matrix for image retrieval __[Matlab Code Here](https://github.com/Zye3/Speckle-Image-Reconstruction/blob/master/calculate_RVITM.m)__

| RVITM         | 0.1665  | 9.8718  |

3. Deep learning for image reconstruction

| Model         | SSIM    | PSNR    |
|---------------|---------|---------|
| FC-AE         | 0.7531  | 12.3864 |
| CNN-AE        | 0.8550  | 25.5408 |
| AE-SNN        | 0.9364  | 16.0765 |
| U-Net         | 0.9368  | 32.7797 |
| Att-U-Net     | 0.9404  | 32.2254 |
| HPM-Att-U-Net | 0.9524  | 33.2440 |





