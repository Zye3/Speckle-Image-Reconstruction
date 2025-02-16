# Import necessary PyTorch libraries
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import argparse
from torchvision.utils import make_grid

from model import ConvAutoencoder, UNet, AutoEncoder, VAE, AE_SNN
from config import parse_args
from data_preprocess import DatasetProcessor
from utils import vis_batch, vae_loss, validate_model, select_model, save_metrics, export_to_onnx

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def current_datetime():
    """
    Get the current date and time for naming purposes.
    """
    return datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')


class TrainingProcessor:
    def __init__(self, data_dir, model_type, code_length, lr, batch_size, epochs, normalize_factor, test_size, val_size, vis=True):
        self.data_dir = data_dir
        self.model_type = model_type
        self.code_length = code_length
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.vis = vis
        self.normalize_factor = normalize_factor
        self.test_size = test_size
        self.val_size = val_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_dir = (f"runs/{model_type}_{self.code_length}_{self.lr}_{self.batch_size}_{self.epochs}_"
                        f"{self.test_size}&{self.val_size}_{current_datetime()}")

    # code_length is the size of the speckle image
    def train_pytorch(self):
        # Initialize TensorBoard
        writer = SummaryWriter(self.log_dir)

        # Initialize the dataset processor with configurations from config.py
        data_processor = DatasetProcessor(data_path=self.data_dir, batch_size=self.batch_size, log_dir=self.log_dir,
                                          normalize_factor=self.normalize_factor, test_size=self.test_size,
                                          val_size=self.val_size)

        # Load datasets
        train_loader, val_loader, test_loader = data_processor.load_dataset()

        # Model selection
        model = select_model(self.model_type, device)
        print(
            f'model_type:{self.model_type}, code_length:{self.code_length}, '
            f'lr:{self.lr}, batch_size:{self.batch_size}, epochs:{self.epochs},\n'
            f'model:{model}')

        # Define optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        # 设定初始的最佳损失为一个很大的值
        best_loss = float('inf')
        train_loss = float('inf')

        # Training loop
        # 1.input to model,
        # 2.output to loss function
        # 3.Resets the gradients of all optimized
        # 4.Backpropagation
        # 5.Performs a single optimization step (parameter update).
        total_time = time.time()
        for epoch in range(self.epochs):
            a = time.time()
            model.train()
            for batch_idx, (data, label) in enumerate(train_loader):
                data, label = data.to(device), label.to(device)
                if self.model_type == 'VAE':
                    output, z_mean, z_log_var = model(data)
                    loss = vae_loss(output, data, z_mean, z_log_var, data.shape[-1])
                # elif self.model_type == 'U-net':
                #     output = model(data)
                #     loss = criterion(output, label)
                else:
                    output = model(data)
                    loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                # Visualize the first batch
                if self.vis and batch_idx == 0 and epoch % 20 == 0:
                    vis_batch(data, label, output, epoch, batch_idx, self.normalize_factor, self.log_dir)

                # Log training loss to TensorBoard
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

            # Validation loop
            avg_val_loss = validate_model(model, self.model_type, val_loader, criterion, device)

            # Log validation loss to TensorBoard
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            print(f'{current_datetime()},Epoch: {epoch}, '
                  f'Train Loss: {loss.item():.8f}, Val Loss: {avg_val_loss:.8f}, '
                  f'Time Costing {time.time() - a:.3f}')

            # 如果当前的验证损失低于最佳损失，则保存模型参数
            if avg_val_loss < best_loss and loss < train_loss:
                train_loss = loss
                best_loss = avg_val_loss
                current_epoch =epoch
                torch.save(model.state_dict(), f'{self.log_dir}/best_model.pth')
                print(f'Saving best model in Epoch:{current_epoch} with loss {train_loss:.8f},{best_loss:.8f}\n')

        TL = time.time() - total_time
        print(f'Total_Time Costing:{TL:.3f}')

        # Save the trained model
        torch.save(model.state_dict(), f'{self.log_dir}/{self.model_type}.pth')


        # Hyperparameter tuning visualization
        writer.add_hparams({'lr': self.lr, 'batch_size': self.batch_size, 'code_length': self.code_length},
                           {'hparam/loss': avg_val_loss})  # Add more metrics as needed

        # Close TensorBoard writer
        writer.close()

        # Save the hyperparameter and Training time
        save_metrics(self.model_type, self.code_length,
                     self.lr, self.batch_size, self.epochs,self.test_size, self.val_size,
                     model, TL, self.log_dir,current_epoch,train_loss,best_loss)

        model = model.to(device)
        return model, self.log_dir


# Main function
if __name__ == '__main__':
    args = parse_args()

    processor = TrainingProcessor(data_dir=args.data_dir, model_type=args.model_type, code_length=args.code_length,
                                  lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
                                  normalize_factor=args.normalize_factor, test_size=args.test_size,
                                  val_size=args.val_size, vis=args.vis)
    model, log_dir= processor.train_pytorch()

    # For a  model, input size might be (1, 1, 256, 256). Adjust based on your model's input.
    export_to_onnx(model, args.batch_size, args.code_length, args.model_type, log_dir, device)

