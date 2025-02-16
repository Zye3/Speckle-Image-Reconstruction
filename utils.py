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


def save_metrics(model_type, code_length, lr, batch_size, epochs,
                 test_size, val_size, model, total_time, log_dir, epoch, train_loss, val_loss):
    """
    Save the train hyperparameter to a text file.
    将超参数保存到一个文本文件中。
    """
    with open(os.path.join(log_dir, 'metrics.txt'), 'w') as f:
        f.write(f'model_type:{model_type}\n'
                f'code_length:{code_length}\n'
                f'lr:{lr}, batch_size:{batch_size}, epochs:{epochs},\n'
                f'best model in Epoch:{epoch} with loss {train_loss:.8f},{val_loss:.8f}\n\n'
                f'test_size:{test_size},val_size:{val_size}\n'
                f'model:{model},\n'
                f'Training Time:{total_time:.3f}')


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


def export_to_onnx(model, batch_size, input_size, model_type, log_dir, device):
    # Set the model to evaluation mode
    model.eval()

    input = (batch_size, 1, input_size, input_size)

    # Create a dummy input tensor appropriate for the model
    dummy_input = torch.randn(input, device=device)

    # Define the filename for the ONNX model
    onnx_file_name = f'{log_dir}/{model_type}.onnx'

    # Export the model to an ONNX file
    torch.onnx.export(model, dummy_input, onnx_file_name, export_params=True, opset_version=10,
                      do_constant_folding=True, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    print(f'Model has been exported to ONNX format and saved to {onnx_file_name}')


def vis_batch(data, label, output, epoch, batch_idx, normalize_factor, log_dir, num_imgs_to_visualize=4):
    """
    Visualize a batch of images during training.
    在训练过程中可视化一批图像。
    参数:
    - data: 输入数据
    - label: 标签数据
    - output: 模型输出
    - batch_idx: 批处理索引
    """
    # Visualization Part
    # if batch_idx % 1 == 0:  # Visualize the first batch

    if batch_idx == 0:  # Visualize the first batch
        fig, axes = plt.subplots(3, num_imgs_to_visualize, figsize=(24, 18))
        for img_index in range(num_imgs_to_visualize):
            # Normalize and display data, label, output images
            norm_data = (normalize_factor * data[img_index]).cpu().detach().squeeze()
            norm_label = (label[img_index]).cpu().detach().squeeze()
            norm_output = (normalize_factor * output[img_index]).cpu().detach().squeeze()

            axes[0, img_index].imshow(norm_data, cmap='gray')
            axes[1, img_index].imshow(norm_label, cmap='gray')
            axes[2, img_index].imshow(norm_output, cmap='gray')

            # Set titles and turn off axis
            axes[0, img_index].set_title(f'Input {img_index}', fontsize=18)
            axes[1, img_index].set_title(f'Label {img_index}', fontsize=18)
            axes[2, img_index].set_title(f'Reconstructed {img_index}', fontsize=18)
            axes[0, img_index].axis('off')
            axes[1, img_index].axis('off')
            axes[2, img_index].axis('off')
        plt.suptitle(f'Visualization for Epoch {epoch}, Batch {batch_idx}', fontsize=18)

        plt.savefig(f'{log_dir}/Epoch_{epoch}_batch_{batch_idx}.png')
