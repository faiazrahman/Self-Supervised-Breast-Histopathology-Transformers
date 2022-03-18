from pathlib import Path
from ctypes import resize
from tkinter.tix import IMAGE
import logging
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

import torch
from torch.distributions.bernoulli import Bernoulli
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms.functional import to_pil_image

import pytorch_lightning as pl

from dataloader import DinoOriginalBreastHistopathologyDataset, DinoPretrainingBreastHistopathologyDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

NUM_CPUS = 40 # 0 | 40

TRAIN_FILES = "data/"
IMAGE_SIZE = 256
PATCH_SIZE = 16
ZERO_PCT = 0.1
PATCHES_PER_ROW = (IMAGE_SIZE // PATCH_SIZE)
NUM_PATCHES = PATCHES_PER_ROW ** 2
RGB_CHANNELS = 3
NUM_PIXELS = PATCH_SIZE ** 2 * RGB_CHANNELS
VALID_IMAGES = 5
TOP_K = 5
NUM_WORKERS = 4

BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

TRANSFORMER_N_HEADS = 8
TRANSFORMER_N_LAYERS = 6

# Update constants
STUDENT_TEMPERATURE = 0.1
TEACHER_TEMPERATURE = 0.05
CENTER_MOMENTUM = 0.9
TEACHER_MOMENTUM = 0.995

class OriginalImageData(Dataset):
    """ Original image data, for validation """

    def __init__(self, image_files):
        self.image_files = image_files
        self.resize = torchvision.transforms.Resize(IMAGE_SIZE, IMAGE_SIZE)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        original_image = torchvision.io.read_image(self.image_files[idx])

        if original_image.shape[0] == 1:
            # Add channels to ensure that all images have 3 channels
            original_image = torch.cat([original_image] * 3)
        resized_output = self.resize(original_image)
        return resized_output

class TrainingImageData(Dataset):
    """
    Transformed image data (with two transformed versions per original image),
    for training via DINO algorithm
    """

    def __init__(self, image_files):
        self.image_files = image_files
        self.transformA = torchvision.transforms.RandomResizedCrop(
            (IMAGE_SIZE, IMAGE_SIZE), scale=(0.5, 1.0))
        self.transformB = torchvision.transforms.RandomResizedCrop(
            (IMAGE_SIZE, IMAGE_SIZE))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Given image is passed through two transforms where one is at leas
        # half as large as the original image
        original_image = torchvision.io.read_image(self.image_files[idx])
        transformed_imageA = self.transformA(original_image)
        transformed_imageB = self.transformB(original_image)

        if original_image.shape[0] == 1:
            # Add channels to ensure that all images have 3 channels
            transformed_imageA = torch.cat([transformed_imageA] * 3)
            transformed_imageB = torch.cat([transformed_imageB] * 3)

        return transformed_imageA, transformed_imageB

class CollateFunction:
    """
    The collate function is used to take the transformed images and divide
    them into patches (of dimension PATCH_SIZE by PATCH_SIZE)
    These patches will be put into a sequence for the Vision Transformer
    """

    def __call__(self, batch):
        transformed_imageA, transformed_imageB = zip(*batch)
        return self.reshape(transformed_imageA), self.reshape(transformed_imageB)

    def reshape(self, batch):
        patches = torch.stack(batch).unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        num_images = len(patches)
        patches = patches.reshape(
            num_images, RGB_CHANNELS, NUM_PATCHES, PATCH_SIZE, PATCH_SIZE)
        patches.transpose_(1, 2)
        reshaped_output = patches.reshape(num_images, NUM_PATCHES, -1) / 255.0 - 0.5
        return reshaped_output

class CollateSingleImage(CollateFunction):
    """ Wrapper for collating a single (batch of) image(s) """
    def __call__(self, batch):
        return self.reshape(batch)

def setup_ssl_dataloaders():
    data_files = [str(file) for file in Path(TRAIN_FILES).glob("*.jpg")]
    train_files, val_files = sklearn.model_selection.train_test_split(
        data_files, test_size=0.15, random_state=6)

    train_data = TrainingImageData(train_files)
    val_data = OriginalImageData(val_files)

    train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=CollateFunction())
    val_loader = DataLoader(val_data, BATCH_SIZE * 2, shuffle=False,
        drop_last=True, num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=CollateSingleImage())

    return train_loader, val_loader

if __name__ == "__main__":
    # train_loader, val_loader = setup_ssl_dataloaders()

    # Get pretraining datasets: training (transformed) and validation (original)
    # Note that the pretraining dataset is the one that is transformed
    full_dataset = DinoPretrainingBreastHistopathologyDataset()
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    # NOTE: You must use the same exact seed for torch.Generate() for both the
    # training and evaluation of a model to ensure that the two datasets have
    # no overlapping examples; otherwise, evaluation will not be truly
    # representative of model performance
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(6)
    )
    final_train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - final_train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [final_train_size, val_size],
        generator=torch.Generator().manual_seed(6)
    )
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE, # TODO args.batch_size,
        num_workers=NUM_CPUS,
        drop_last=True,
        collate_fn=CollateFunction()
    )
    logging.info(train_loader)

    # Note that by maintaining the same seed values, we ensure no overlap
    # See note above about torch.utils.data.random_split
    full_dataset = DinoOriginalBreastHistopathologyDataset()
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(6)
    )
    final_train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - final_train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [final_train_size, val_size],
        generator=torch.Generator().manual_seed(6)
    )
    logging.info(f"Val dataset size: {len(val_dataset)}")
    logging.info(test_dataset)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE, # TODO args.batch_size,
        num_workers=NUM_CPUS,
        drop_last=True,
        collate_fn=CollateSingleImage()
    )
    logging.info(val_loader)

    # TODO: rm
    x, y = next(iter(train_loader))
    x2 = next(iter(val_loader))
    print(x, y)
    print(x2)

