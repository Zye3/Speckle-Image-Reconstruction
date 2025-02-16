import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os


class DatasetProcessor:
    def __init__(self, data_path, batch_size, log_dir, normalize_factor, test_size, val_size):
        self.data_path = data_path
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.normalize_factor = normalize_factor
        self.test_size = test_size
        self.val_size = val_size
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def plot_sample_images(self, speckle_images, original_images, index, log_dir, name:str, figsize=(12, 6), cmap='gray'):
        """
        Plots and saves a comparison of speckle and original images.

        Parameters:
            speckle_images (np.array): The speckle images array.
            original_images (np.array): The original images array.
            index (int): The index of the images to display.
            log_dir (str): Directory where to save the image.
            figsize (tuple): Size of the figure (width, height).
            cmap (str): Colormap for displaying images.
        """

        # print(
        #     f'Plot Sample before training:{speckle_images[index].shape}-> {np.squeeze(speckle_images[index]).shape}\n')
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        im1 = axes[0].imshow(np.squeeze(speckle_images[index]), cmap=cmap)
        axes[0].set_title(f'Speckle Image of {name}')
        axes[0].axis('off')

        im2 = axes[1].imshow(np.squeeze(original_images[index]), cmap=cmap)
        axes[1].set_title(f'Original Image of {name}')
        axes[1].axis('off')

        fig.savefig(f'{log_dir}/before_train_image_of_{name}.png')

    def split_data(self, X, Y, index, test_size, val_size):
        """
        Splits data into training, validation, and test sets sequentially.

        Parameters:
            X (np.array): Features dataset.
            Y (np.array): Labels dataset.
            index (List): the len of dataset
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the dataset to include in the validation split.

        Returns:
            X_train, X_val, X_test, Y_train, Y_val, Y_test (tuple): Training, validation, and test sets.
        """
        total_samples = len(index)
        test_count = int(total_samples * test_size)
        val_count = int(total_samples * val_size)

        # Calculate the ending index of the training data
        train_end = total_samples - val_count - test_count

        # Sequentially split the datasets
        X_train, X_val, X_test = X[:train_end], X[train_end:train_end + val_count], X[train_end + val_count:]
        Y_train, Y_val, Y_test = Y[:train_end], Y[train_end:train_end + val_count], Y[train_end + val_count:]

        return X_train, X_val, X_test, Y_train, Y_val, Y_test

    def process_images(self, speckle, original, index):
        """
        Processes and normalizes speckle and original images.
        [H,W] -> [C,H,W] with range [0,1]!!! Don't normalize to [-1,1]
        Parameters:
            speckle (np.array): The raw speckle images.[H,W]
            original (np.array): The raw original images.[H,W]
            index (np.array): The indices of images to process.
            Normalize_factor(Int): The factor of Normalization.[H,W]
        Returns:
            speckle_images, original_images (tuple): Normalized and processed images.[C,H,W]
        """
        speckle_images = []
        original_images = []
        # Normalize
        for i in range(len(index)):
            speckle_image = np.expand_dims(speckle[index[i], :, :] / self.normalize_factor, axis=0).astype(
                np.float32)  # 2^16
            original_image = np.expand_dims(original[index[i], :, :], axis=0).astype(np.float32)

            speckle_images.append(speckle_image)
            original_images.append(original_image)

        # print(f'speckle_image.shape:{speckle_images[1], type(speckle_images)}')
        # print(f'original_image.shape:{original_images[1], type(speckle_images)}')

        speckle_images = np.array(speckle_images)  # Convert list to array
        original_images = np.array(original_images)  # Convert list to array

        # print(f'After speckle_image.shape:{speckle_images.shape, type(speckle_images)}')
        # print(f'After original_image.shape:{original_images.shape, type(original_images)}\n')
        return speckle_images, original_images

    # Function to load and preprocess the dataset
    def load_dataset(self):
        """
        Loads and preprocesses the dataset from provided paths and creates data loaders for training, validation, and testing sets.

        Parameters:
            data_path (list): List of paths for speckle images, original images, and index files respectively.
            batch_size (int): The size of each batch during training/testing.
            log_dir (str): The directory where logs and sample images are saved.

        Returns:
            train_loader, val_loader, test_loader (tuple): Data loaders for the training, validation, and test sets.
        """
        speckle = np.load(self.data_path[0])  # speckle image = input size 256x256
        original = np.load(self.data_path[1])  # original image = label 128x128/256x256
        index = np.load(self.data_path[2])  # index/32768
        # Normalization and add a dim at first channel
        speckle_images, original_images = self.process_images(speckle, original, index)
        # Split dataset into training, validation, and test sets using the custom function
        X_train, X_val, X_test, Y_train, Y_val, Y_test = self.split_data(
            speckle_images, original_images, index, test_size=self.test_size, val_size=self.val_size)

        # Using else dataset as validation dataset
        if len(self.data_path) > 3:
            speckle_X_val = np.load(self.data_path[3])  # load another dataset as validation
            original_Y_val = np.load(self.data_path[4])
            index_X_val = np.load(self.data_path[5])

            speckle_X_test = np.load(self.data_path[3])  # load another dataset as test
            original_Y_test = np.load(self.data_path[4])
            index_X_test = np.load(self.data_path[5])

            # Normalization and add a dim at first channel
            speckle_images_val, original_images_val = self.process_images(speckle_X_val, original_Y_val, index_X_val)
            speckle_images_test, original_images_test = self.process_images(speckle_X_test, original_Y_test, index_X_test)

            # Split dataset into training, validation, and test sets using the custom function
            X_train, X_val, X_test, Y_train, Y_val, Y_test = self.split_data(
                speckle_images, original_images, index, test_size=0, val_size=0)

            X_val, X_test, Y_val, Y_test = speckle_images_val, original_images_val, speckle_images_test, original_images_test


        # Display the train validation and test image before train
        self.plot_sample_images(X_train, Y_train, 1, self.log_dir,name='train')
        self.plot_sample_images(X_val, Y_val, 2, self.log_dir,name='validation')
        self.plot_sample_images(X_test, Y_test, 3, self.log_dir,name='test')

        # Convert np.array to tensors, and then to TensorDataset
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test))

        # Create DataLoader instances, it will add a batch dim [16,1,256,256]
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        # Print the shape of the datasets
        print(f'Train DataLoader batch shape: {next(iter(train_loader))[0].shape}')
        print(f'Validation DataLoader batch shape: {next(iter(val_loader))[0].shape}')
        print(f'Test DataLoader batch shape: {next(iter(test_loader))[0].shape}\n')

        return train_loader, val_loader, test_loader

