import time

import torch
from torchvision import transforms
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from data_preprocess import DatasetProcessor
from config import parse_args
from utils import select_model
import datetime
import os
import matplotlib.pyplot as plt
import cv2
import csv


def current_datetime():
    """
    Get the current date and time for naming purposes.
    """
    return datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')


def load_model(model_path, model_type, device):
    """ Load the saved model from a given path """
    model = select_model(model_type, device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def save_metrics(avg_ssim, avg_psnr, log_dir):
    """
    Save the average SSIM and PSNR to a text file.
    将平均SSIM和PSNR保存到一个文本文件中。
    """
    with open(os.path.join(log_dir, '1.metrics.txt'), 'w') as f:
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")

def save_to_csv(data_rows, filename, header):
    """
    Save data to a CSV file.

    Parameters:
    - data_rows: List of rows (each row is a list of values) to be written to the CSV file.
    - filename: Full path to the file where data will be saved.
    - header: List of column headers.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header first
        writer.writerows(data_rows)  # Write data rows


def vis(data, label, output, ssim_single, psnr_single, batch_idx,img_index, normalize_factor, log_dir):
    """
    Visualize the input, label, and output images in separate figures.
    分别在不同图中可视化输入、标签和输出图像。
    """
    # for img_index in range(data.size(0)):
    # Create a figure for each set of images
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Normalize and display data, label, output images
    # norm_data = (normalize_factor * data[img_index]).cpu().detach().squeeze()
    # norm_label = (label[img_index]).cpu().detach().squeeze()
    # norm_output = (normalize_factor * output[img_index]).cpu().detach().squeeze()
    norm_data = data
    norm_label = label
    norm_output = output
    axes[0].imshow(norm_data, cmap='gray')
    axes[1].imshow(norm_label, cmap='gray')
    axes[2].imshow(norm_output, cmap='gray')

    # Set titles and turn off axis
    axes[0].set_title(f'Input', fontsize=18)
    axes[1].set_title(f'Label', fontsize=18)
    axes[2].set_title(f'Reconstructed', fontsize=18)
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')

    plt.suptitle(f'SSIM {ssim_single:.3f}& PSNR{psnr_single:.3f}', fontsize=18)
    plt.savefig(f'{log_dir}/Image_{batch_idx}_{img_index}.png')
    plt.close(fig)

    # Create a figure for output images
    cv2.imwrite(f'{log_dir}/Image_{batch_idx}_{img_index}_output.png', norm_output*normalize_factor)


def evaluate_model(model, model_type, dataloader, device, normalize_factor, log_dir, visualize=True):
    """ Evaluate the model using SSIM and PSNR metrics and visualize results """
    ssim_total = 0.0
    psnr_total = 0.0
    num_samples = 0
    metrics_list = []  # List to store all metrics for CSV

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            a = time.time()
            data, target = data.to(device), target.to(device)
            if model_type == 'VAE':
                output, z_mean, z_log_var = model(data)
            else:
                output = model(data)
            # print(data.shape, target.shape, output.shape) # data = Input, target = label/GT, output = prediction
            b = time.time()-a
            print(b)
            # Convert tensors to numpy arrays for metric calculation
            label_np = target.cpu().numpy()
            output_np = output.cpu().numpy()
            input_np = data.cpu().numpy()

            for i in range(data.size(0)):
                label = np.squeeze(label_np[i])
                reconstructed = np.squeeze(output_np[i])
                input = np.squeeze(input_np[i])

                ssim_single = ssim(label, reconstructed, data_range=label.max() - label.min())
                psnr_single = psnr(label, reconstructed, data_range=label.max() - label.min())

                # Append each batch's metrics to the list
                metrics_list.append([batch_idx, i, ssim_single, psnr_single])

                # Visualization of each batch using vis
                if visualize:
                    vis(input, label, reconstructed, ssim_single, psnr_single, batch_idx, i, normalize_factor, log_dir)

                ssim_total += ssim_single
                psnr_total += psnr_single
                num_samples += 1

    avg_ssim = ssim_total / num_samples
    avg_psnr = psnr_total / num_samples

    # Save all metrics to CSV file
    metrics_file = os.path.join(log_dir.split('/')[0], log_dir.split('/')[1],
                                f'1.ssim&psnr',
                                f'{model_type}_ssim_psnr.csv')
    save_to_csv(metrics_list, metrics_file, ['Batch Index', 'Image Index', 'SSIM', 'PSNR'])
    save_metrics(avg_ssim, avg_psnr, log_dir)  # Save the metrics to a text file
    return avg_ssim, avg_psnr


def main():
    args = parse_args()

    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model !!! Note: replace the mode_type in the config,py first
    model_path = (f'runs/'
                  f'Multi_64_64_0.2_0.1/'
                  f'Att_UNet_64_0.001_48_90_0.1&0.2_2024-08-15__21-28-29/'
                  f'{args.model_type}.pth')
    model = load_model(model_path, args.model_type, device)

    # load log_dir
    log_dir = f"test/{model_path.split('/')[1]}/{model_path.split('/')[2]}/new_3"
    # Load the dataset
    data_processor = DatasetProcessor(data_path=args.data_dir, batch_size=args.batch_size, log_dir=log_dir,
                                      normalize_factor=args.normalize_factor,
                                      test_size=args.test_size, val_size=args.val_size)
    _, _, test_loader = data_processor.load_dataset()  # Note: already normalize the input

    # Evaluate the model
    avg_ssim, avg_psnr = evaluate_model(model, args.model_type, test_loader, device, args.normalize_factor, log_dir)
    print(f"Average SSIM: {avg_ssim:.4f}, Average PSNR: {avg_psnr:.4f}")


if __name__ == "__main__":
    main()
