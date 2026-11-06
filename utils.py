import os

import matplotlib.pyplot as plt
import torch
from model import ConvAutoencoder, UNet, AutoEncoder, AE_SNN, Att_UNet, HPM_Attention_UNet


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
    elif model_type == 'Att_UNet':
        return Att_UNet().to(device)
    elif model_type == 'HPM_UNet':
        return HPM_Attention_UNet().to(device)
    else:
        raise ValueError("Unsupported model type")


def mse_loss(recon_x, x):
    return torch.mean((x - recon_x) ** 2)


def _as_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def vis_batch(data, label, output, epoch, batch_idx, normalize_factor, log_dir):
    """Save the first sample of a batch: input, label, reconstruction."""
    os.makedirs(log_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = (
        ('Input', data),
        ('Label', label),
        ('Reconstructed', output),
    )
    for ax, (title, tensor) in zip(axes, panels):
        image = tensor[0].detach().float().cpu().squeeze().numpy()
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(f'epoch {epoch} batch {batch_idx}')
    fig.savefig(os.path.join(log_dir, f'vis_e{epoch}_b{batch_idx}.png'))
    plt.close(fig)


def save_metrics(model_type, code_length, lr, batch_size, epochs, test_size, val_size,
                 model, total_time, log_dir, current_epoch, train_loss, best_loss):
    """Write the training summary next to the checkpoint."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, 'train_metrics.txt')
    with open(path, 'w') as handle:
        handle.write(
            f'model_type: {model_type}\n'
            f'code_length: {code_length}\n'
            f'lr: {lr}\n'
            f'batch_size: {batch_size}\n'
            f'epochs: {epochs}\n'
            f'test_size: {test_size}\n'
            f'val_size: {val_size}\n'
            f'best_epoch: {current_epoch}\n'
            f'train_loss: {_as_float(train_loss)}\n'
            f'val_loss: {_as_float(best_loss)}\n'
            f'seconds: {total_time:.3f}\n'
        )
    print(f'wrote {path}')


def export_to_onnx(model, batch_size, code_length, model_type, log_dir, device):
    """Export one sample. A missing ONNX package does not fail training."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f'{model_type}.onnx')
    sample = torch.rand(1, 1, int(code_length), int(code_length), device=device)
    model.eval()
    try:
        torch.onnx.export(
            model, sample, path,
            input_names=['input'], output_names=['output'],
        )
        print(f'wrote {path}')
    except Exception as exc:
        print(f'onnx export skipped: {type(exc).__name__}: {exc}')


def validate_model(model, model_type, val_loader, criterion, device):
    total_val_loss = 0
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)

            output = model(data)
            val_loss = mse_loss(output, label)

            if torch.isnan(output).any():
                print("NaN detected")

            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    return avg_val_loss


