import json
import math
import multiprocessing
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, UnidentifiedImageError, ImageFile
from cjm_pandas_utils.core import markdown_to_pandas
from cjm_pytorch_utils.core import set_seed, get_torch_device
from sklearn.model_selection import train_test_split
from timm.models import resnet
from torch import Tensor
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics import MulticlassAccuracy
from torchtnt.utils import get_module_summary
from torchvision import transforms
from tqdm.auto import tqdm

from app.AI_Python import ResizePad

data_path = 'C:\\Users\\Administrator\\Downloads\\bag_image'

device = get_torch_device()


class CustomImageDataset(Dataset):
    def __init__(self, image_files, class_map, transform=None, target_transform=None):
        self.img_labels = image_files
        self.class_map = class_map
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels[idx]
        try:
            image = np.array(Image.open(img_path).convert('RGB'))

            label = '|'.join(img_path.split('\\')[5:-1])
            label_int = self.class_map[label]
            if self.transform:
                image = self.transform(image)
        except UnidentifiedImageError:
            print(f'Failed to read image: {img_path}')
            return None

        return image, label_int


class CustomTrivialAugmentWide(transforms.TrivialAugmentWide):
    # The _augmentation_space method defines a custom augmentation space for the augmentation policy.
    # This method returns a dictionary where each key is the name of an augmentation operation and
    # the corresponding value is a tuple of a tensor and a boolean value.
    # The tensor defines the magnitude of the operation, and the boolean defines
    # whether to perform the operation in both the positive and negative directions (True)
    # or only in the positive direction (False).
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        # Define custom augmentation space
        custom_augmentation_space = {
            # Identity operation doesn't change the image
            "Identity": (torch.tensor(0.0), False),

            # Distort the image along the x or y-axis, respectively.
            "ShearX": (torch.linspace(0.0, 0.25, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.25, num_bins), True),

            # Move the image along the x or y-axis, respectively.
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),

            # Rotate operation: rotates the image.
            "Rotate": (torch.linspace(0.0, 45.0, num_bins), True),

            # Adjust brightness, color, contrast,and sharpness respectively.
            "Brightness": (torch.linspace(0.0, 0.75, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),

            # Reduce the number of bits used to express the color in each channel of the image.
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),

            # Invert all pixel values above a threshold.
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),

            # Maximize the image contrast by setting the darkest color to black and the lightest to white.
            "AutoContrast": (torch.tensor(0.0), False),

            # Equalize the image histogram to improve its contrast.
            "Equalize": (torch.tensor(0.0), False),
        }

        # The function returns the dictionary of operations.
        return custom_augmentation_space


def list_nested_files(directory):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            files.append(file_path)
    return files


def generate_label():
    image_files = []
    class_labels = []

    all_files = list_nested_files(data_path)
    for file in all_files:
        parts = file.split('\\')
        image_files.append('\\'.join(parts))
        class_labels.append('|'.join(parts[5:-1]))

    cc = pd.DataFrame(class_labels)
    grouped = cc.groupby(cc.columns[0]).size().sort_values(ascending=False)
    grouped.to_csv('grouping.csv')

    class_names = sorted(list(set(class_labels)))
    class_map = dict(zip(class_names, range(len(class_names))))

    classification_map = pd.DataFrame.from_dict(class_map, orient='index')
    classification_map.columns = ['label_int']
    classification_map.to_csv('labels.csv')

    return image_files, class_labels, class_names, class_map


def build_model(class_names):
    # Define the ResNet model variant to use
    resnet_model = 'resnet50d'  # resnet152d.ra2_in1k

    # Get the default configuration of the chosen model
    model_cfg = resnet.default_cfgs[resnet_model].default.to_dict()
    print(pd.DataFrame.from_dict(model_cfg, orient='index'))

    mean, std = model_cfg['mean'], model_cfg['std']
    norm_stats = (mean, std)

    resnet152 = timm.create_model(resnet_model, pretrained=True, num_classes=len(class_names))
    # Set the device and data type for the model
    resnet152 = resnet152.to(device=device, dtype=torch.float32)
    # Add attributes to store the device and model name for later reference
    resnet152.device = device
    resnet152.name = resnet_model

    # Define the input to the model
    test_inp = torch.randn(1, 3, 256, 256).to(device)

    # Get a summary of the model as a Pandas DataFrame
    summary_df = markdown_to_pandas(f"{get_module_summary(resnet152, [test_inp])}")
    # Filter the summary to only contain Conv2d layers and the model
    summary_df = summary_df[(summary_df.index == 0) | (summary_df['Type'] == 'Conv2d')]
    # Remove the column "Contains Uninitialized Parameters?"
    final_df = summary_df.drop('Contains Uninitialized Parameters?', axis=1)
    final_df.to_csv('network.csv')

    return model_cfg, norm_stats, resnet152


def train_val_dataset(image_files, class_labels, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(image_files))), test_size=val_split, stratify=class_labels)
    train_images = [image_files[i] for i in train_idx]
    val_images = [image_files[i] for i in val_idx]
    return train_images, val_images


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def generate_dataset(norm_stats, image_files, class_labels, class_map):
    train_sz = (288, 288)
    trivial_aug = CustomTrivialAugmentWide()
    train_tfms = transforms.Compose([
        transforms.ToPILImage(),
        ResizePad(max_sz=max(train_sz)),
        trivial_aug,
        transforms.ToTensor(),
        transforms.Normalize(*norm_stats),
    ])

    valid_tfms = transforms.Compose([
        transforms.ToPILImage(),
        ResizePad(max_sz=max(train_sz)),
        transforms.ToTensor(),
        transforms.Normalize(*norm_stats),
    ])

    train_split, val_split = train_val_dataset(image_files, class_labels, val_split=0.2)

    # Instantiate the datasets using the defined transformations
    train_dataset = CustomImageDataset(train_split, class_map=class_map, transform=train_tfms)
    valid_dataset = CustomImageDataset(val_split, class_map=class_map, transform=valid_tfms)

    # Print the number of samples in the training and validation datasets
    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(valid_dataset)}')
    print(f'Number of image: {len(image_files)}')

    # Set the number of worker processes for loading data. This should be the number of CPUs available.
    num_workers = multiprocessing.cpu_count()

    # Define parameters for DataLoader
    data_loader_params = {
        'batch_size': 32,  # Batch size for data loading
        'num_workers': num_workers,  # Number of subprocesses to use for data loading
        'persistent_workers': True,
        # If True, the data loader will not shut down the worker processes after a dataset has been consumed once.
        # This allows to maintain the worker dataset instances alive.
        'pin_memory': True,
        # If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Useful when using
        # GPU.
        'pin_memory_device': device,
        # Specifies the device where the data should be loaded. Commonly set to use the GPU.
    }

    # Create DataLoader for training data. Data is shuffled for every epoch.
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, **data_loader_params, shuffle=True)

    # Create DataLoader for validation data. Shuffling is not necessary for validation data.
    valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, **data_loader_params)

    # Print the number of batches in the training and validation DataLoaders
    print(f'Number of batches in train DataLoader: {len(train_dataloader)}')
    print(f'Number of batches in validation DataLoader: {len(valid_dataloader)}')

    return train_dataloader, valid_dataloader


# Function to run a single training/validation epoch
def run_epoch(model, dataloader, optimizer, metric, lr_scheduler, device, scaler, epoch, is_training):
    # Set model to training mode if 'is_training' is True, else set to evaluation mode
    model.train() if is_training else model.eval()

    # Reset the performance metric
    metric.reset()
    # Initialize the average loss for the current epoch
    epoch_loss = 0
    # Initialize progress bar with total number of batches in the dataloader
    progress_bar = tqdm(total=len(dataloader), desc="Train" if is_training else "Eval")

    # Iterate over data batches
    for batch_id, (inputs, targets) in enumerate(dataloader):
        # Move inputs and targets to the specified device (e.g., GPU)
        inputs, targets = inputs.to(device), targets.to(device)

        # Enables gradient calculation if 'is_training' is True
        with torch.set_grad_enabled(is_training):
            # Automatic Mixed Precision (AMP) context manager for improved performance
            with autocast(device):
                outputs = model(inputs)  # Forward pass
                loss = torch.nn.functional.cross_entropy(outputs, targets)  # Compute loss

        # Update the performance metric
        metric.update(outputs.detach().cpu(), targets.detach().cpu())

        # If in training mode
        if is_training:
            if scaler is not None:  # If using AMP
                # Scale the loss and backward propagation
                scaler.scale(loss).backward()
                scaler.step(optimizer)  # Make an optimizer step
                scaler.update()  # Update the scaler
            else:
                loss.backward()  # Backward propagation
                optimizer.step()  # Make an optimizer step

            optimizer.zero_grad()  # Clear the gradients
            lr_scheduler.step()  # Update learning rate

        loss_item = loss.item()
        epoch_loss += loss_item

        # Update progress bar
        progress_bar.set_postfix(accuracy=metric.compute().item(),
                                 loss=loss_item,
                                 avg_loss=epoch_loss / (batch_id + 1),
                                 lr=lr_scheduler.get_last_lr()[0] if is_training else "")
        progress_bar.update()

        # If loss is NaN or infinity, stop training
        if math.isnan(loss_item) or math.isinf(loss_item):
            print(f"Loss is NaN or infinite at epoch {epoch}, batch {batch_id}. Stopping training.")

    progress_bar.close()
    return epoch_loss / (batch_id + 1)


# Main training loop
def train_loop(model,
               train_dataloader,
               valid_dataloader,
               optimizer,
               metric,
               lr_scheduler,
               device,
               epochs,
               use_amp,
               checkpoint_path,
               model_path):
    # Initialize GradScaler for Automatic Mixed Precision (AMP) if 'use_amp' is True
    scaler = GradScaler() if use_amp else None
    best_loss = float('inf')

    train_loss_list = []
    valid_loss_list = []

    # Iterate over each epoch
    for epoch in tqdm(range(epochs), desc="Epochs"):
        print(f'Running Epoch - {epoch + 1}...')
        # Run training epoch and compute training loss
        train_loss = run_epoch(model,
                               train_dataloader,
                               optimizer,
                               metric,
                               lr_scheduler,
                               device,
                               scaler,
                               epoch,
                               is_training=True)
        train_loss_list.append(train_loss)

        with torch.no_grad():
            # Run validation epoch and compute validation loss
            valid_loss = run_epoch(model,
                                   valid_dataloader,
                                   None,
                                   metric,
                                   None,
                                   device,
                                   scaler,
                                   epoch,
                                   is_training=False)
        valid_loss_list.append(valid_loss)

        # If current validation loss is lower than the best one so far, save model and update best loss
        if valid_loss < best_loss:
            best_loss = valid_loss
            metric_value = metric.compute().item()
            torch.save(model.state_dict(), checkpoint_path)

            training_metadata = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'accuracy': metric_value,
                'learning_rate': lr_scheduler.get_last_lr()[0],
                'model_architecture': model.name
            }
            print(training_metadata)

            # Save best_loss and metric_value in a JSON file
            with open('model/resnet50d/classification_training_metadata.json', 'w') as f:
                json.dump(training_metadata, f)

        # If loss is NaN or infinity, stop training
        if any(math.isnan(loss) or math.isinf(loss) for loss in [train_loss, valid_loss]):
            print(f"Loss is NaN or infinite at epoch {epoch}. Stopping training.")
            break

    # save entire model
    torch.save(model, model_path)
    # If using AMP, clean up the unused memory in GPU
    if use_amp:
        torch.cuda.empty_cache()
    return train_loss_list, valid_loss_list


if __name__ == "__main__":
    # Set options for Pandas DataFrame display
    pd.set_option('max_colwidth', None)  # Do not truncate the contents of cells in the DataFrame
    pd.set_option('display.max_rows', None)  # Display all rows in the DataFrame
    pd.set_option('display.max_columns', None)  # Display all columns in the DataFrame
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    seed = 1234
    set_seed(seed)

    image_files, class_labels, class_names, class_map = generate_label()
    model_cfg, norm_stats, resnet152 = build_model(class_names)
    train_dataloader, valid_dataloader = generate_dataset(norm_stats, image_files, class_labels, class_map)

    # The model checkpoint path
    checkpoint_dir = ''
    # The model checkpoint path
    checkpoint_path = checkpoint_dir + "handbag_classfication_weights.pth"
    model_path = checkpoint_dir + "handbag_classfication.pt"

    # Learning rate for the model
    lr = 1e-3

    # Number of training epochs
    epochs = 38

    # AdamW optimizer; includes weight decay for regularization
    optimizer = torch.optim.AdamW(resnet152.parameters(), lr=lr, eps=1e-5)

    # Learning rate scheduler; adjusts the learning rate during training
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                       max_lr=lr,
                                                       total_steps=epochs * len(train_dataloader))

    # Performance metric: Multiclass Accuracy
    metric = MulticlassAccuracy()

    # Check for CUDA-capable GPU availability
    use_amp = torch.cuda.is_available()
    train_loss_list, valid_loss_list = train_loop(resnet152,
                                                  train_dataloader,
                                                  valid_dataloader,
                                                  optimizer,
                                                  metric,
                                                  lr_scheduler,
                                                  device,
                                                  epochs,
                                                  use_amp,
                                                  checkpoint_path,
                                                  model_path)

    print(train_loss_list)
    print(valid_loss_list)
