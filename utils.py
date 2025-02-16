import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
from model import ConvAutoencoder, UNet, AutoEncoder, VAE, AE_SNN, Att_UNet, HPM_Attention_UNet
import os


def optimizer():
    return

def select_model(model_type, device):
    """
    Select the model based on the specified type.
    根据指定的类型选择模型。
    返回:
    - 选定的模型实例
    """
    if model_type == 'CNN-AE':
        return ConvAutoencoder().to(device)
    elif model_type == 'U-net':
        return UNet().to(device)
    elif model_type == 'AE':
        return AutoEncoder().to(device)
    elif model_type == 'AE-SNN':
        return AE_SNN().to(device)
    elif model_type == 'VAE':
        return VAE().to(device)
    elif model_type == 'Att_UNet':
        return Att_UNet().to(device)
    elif model_type == 'HPM_UNet':
        return HPM_Attention_UNet().to(device)
    else:
        raise ValueError("Unsupported model type")


def mse_loss(recon_x, x):
    return torch.mean((x - recon_x) ** 2)


def ssim_loss(recon_x, x):
    # Ensure the input tensors are floating point numbers
    x = x.float()
    recon_x = recon_x.float()

    # Compute SSIM over the batch of images.
    # ssim() returns values between -1 and 1, where 1 means perfect similarity.
    # Therefore, we subtract the result from 1 to interpret the loss correctly:
    # the lower the ssim_loss, the more similar y_pred is to y_true.
    return 1.0 - ssim(x, recon_x, data_range=1.0, size_average=True)  # Ensure to match the data range if it's different


def vae_loss(recon_x, x, z_mean, z_log_var, image_size, beta=10):
    """
    Compute the VAE loss function.
    The first term represents the reconstruction likelihood and the other term ensures that our learned distribution q
    is similar to the true prior distribution p.
    Thus our total loss consists of two terms, one is reconstruction error and the other is KL-divergence loss:
    Loss = L\left( {x, \hat x} \right) + \sum\limits_j {KL\left( {{q_j}\left( {z|x} \right)||p\left( z \right)} \right)}

    Parameters:
    - recon_x: reconstructed images.
    - x: original images.
    - z_mean: mean from the latent space.
    - z_log_var: log variance from the latent space.
    - image_size: the height/width of the images (assumed to be square).z
    - beta: weighting factor for the KL divergence.

    Returns:
    - Total loss, reconstruction loss, and KL divergence loss.
    """
    # print(f"input image:{recon_x}\n"
    #       f"label:{x}")
    # Reconstruction loss (assuming the input is normalized to [0,1])
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # Scale the BCE loss to the same scale as the TensorFlow example
    BCE = BCE * (image_size * image_size)
    # KL divergence
    KL_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    # Total loss with the weighted KL divergence
    total_loss = BCE + beta * KL_div
    # print("total_loss, BCE,KL:",total_loss,BCE,KL_div)

    return total_loss


def validate_model(model, model_type, val_loader, criterion, device):
    total_val_loss = 0
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)

            if model_type == 'VAE':
                output, z_mean, z_log_var = model(data)
                val_loss = vae_loss(output, data, z_mean, z_log_var, data.shape[-1])
            # elif model_type == 'U-net':
            #     output = model(data)
            #     val_loss = mse_loss(output, label)
            else:
                output = model(data)
                val_loss = mse_loss(output, label)

            if torch.isnan(output).any():
                print("NaN detected")

            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    return avg_val_loss


