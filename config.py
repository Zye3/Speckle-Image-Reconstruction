import numpy as np
import argparse
import os


def parse_args():
    # Absolute path to the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
            # 8.Speckle_64_64_Digits , 7.Speckle_64_64
    parser = argparse.ArgumentParser(description='Train a model on speckle image reconstruction.')
    parser.add_argument('--data_dir', nargs=3, type=str,
                        default=[os.path.join(script_dir.split('2.')[0], "1.Data", "7.Speckle_64_64", "speckles_64.npy"),
                                 os.path.join(script_dir.split('2.')[0], "1.Data", "7.Speckle_64_64", "label_64.npy"),
                                 os.path.join(script_dir.split('2.')[0], "1.Data", "7.Speckle_64_64", "index.npy")],
                        help='Directory where datasets are saved.')

    parser.add_argument('--model_type', type=str,
                        default='Att_UNet',
                        choices=['CNN-AE', 'U-net', 'AE', 'VAE', 'AE-SNN', 'Att_UNet', 'HPM_UNet'],  # CNN-AE (in,256) (out,128)
                        help='Type of model to train. Default is VAE.')

    parser.add_argument('--code_length', type=int,
                        default=64,
                        help='Channel of code (2 means 2x16 = 32, the output size is 32xHxW and input size is 64xHxW'
                             '2 = 64x64 and 32x32'
                             '16 = 256x256 and 256x256).')

    parser.add_argument('--lr', type=float,
                        default=1e-3,
                        help='Learning rate.')

    parser.add_argument('--batch_size', type=int,
                        default=128,
                        help='Batch size.')

    parser.add_argument('--epochs', type=int,
                        default=90,
                        help='Number of epochs.')

    parser.add_argument('--normalize_factor', type=int,
                        default=np.power(2, 8),
                        help='Normalization the image range to [0,1]. [H,W] 65536')

    parser.add_argument('--test_size', type=float,
                        default=0.2,
                        help='The percentage of test images to use.')

    parser.add_argument('--val_size', type=float,
                        default=0.1,
                        help='The percentage of validation images to use.')

    parser.add_argument('--vis', type=bool,
                        default=True,
                        help='Visualization.')

    return parser.parse_args()

