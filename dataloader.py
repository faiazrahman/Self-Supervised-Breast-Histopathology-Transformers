import sys
import os
import re
from pathlib import Path
import logging
import argparse
import enum

import pandas as pd
from tqdm import tqdm
import pickle

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

DATA_PATH = "./data"
PL_ASSETS_PATH = "./lightning_logs"
IMAGE_EXTENSION = ".png"
PRESAVED_IMAGE_FILEPATHS = "image_filepaths.pkl"
DINO_IMAGE_SIZE = 224

PRETRAINING_IMAGE_SIZE = 256

class BreastHistopathologyDataset(Dataset):
    """ Dataset used by ResNetModel (and any other baseline models) for IDC classification """

    def __init__(self, force_reset=False, image_dim=30):

        self.image_dim = image_dim

        # Get a list of all image files, across all patients
        image_filepaths = list()
        presaved_image_filepaths_path = os.path.join(DATA_PATH, PRESAVED_IMAGE_FILEPATHS)
        if os.path.exists(presaved_image_filepaths_path) and not force_reset:
            # Load from previous run
            logging.info(f"Loading list of image file paths from previous run (stored in {presaved_image_filepaths_path})...")
            logging.info("  To run from scratch, pass in force_reset=True to BreastHistopathologyDataset()")
            with open(presaved_image_filepaths_path, "rb") as file:
                image_filepaths = pickle.load(file)
        else:
            # Get all patient IDs, which are numbers (which we match for via regex)
            # Note that each patient has their own subdirectory in `data/`
            p = re.compile("^\d+$")
            patient_ids = [dir for dir in os.listdir(DATA_PATH) if p.match(dir)]
            logging.info(f"Number of patients: {len(patient_ids)}")

            # Iterate through all patient subdirectories and get filenames
            for patient_id in patient_ids:
                patient_dir = os.path.join(DATA_PATH, patient_id)
                for root, dirs, files in os.walk(patient_dir):
                    curr_filepaths = [os.path.join(root, filename) for filename in files]
                    image_filepaths.extend(curr_filepaths)

            # Save for future runs
            with open(presaved_image_filepaths_path, "wb") as file:
                pickle.dump(image_filepaths, file)

        # Extract the label for each image file
        image_labels = [0 for _ in range(len(image_filepaths))]
        for idx, filename in enumerate(image_filepaths):
            # Files are named as <patient_id>_idx..._x..._y..._class<label>.png
            # We extract the label from that filename
            label = int(filename.replace(IMAGE_EXTENSION, "")[-1])
            image_labels[idx] = label

        self.dataframe = pd.DataFrame(
            list(zip(image_filepaths, image_labels)),
            columns=["image_filepath", "label"]
        )

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, idx):

        image_filepath = self.dataframe.loc[idx, "image_filepath"]
        label = self.dataframe.loc[idx, "label"]

        image = Image.open(image_filepath).convert("RGB")
        image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(self.image_dim, self.image_dim)),
            torchvision.transforms.ToTensor(),
            # All torchvision models expect the same normalization mean and std
            # https://pytorch.org/docs/stable/torchvision/models.html
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])
        image = image_transform(image)

        item = {
            "image_path": image_filepath,
            "image": image,
            "label": label,
        }
        return item

class DinoBreastHistopathologyDataset(Dataset):
    """ Dataset used by DINO models for IDC classification """

    def __init__(self, force_reset=False):

        # Get a list of all image files, across all patients
        image_filepaths = list()
        presaved_image_filepaths_path = os.path.join(DATA_PATH, PRESAVED_IMAGE_FILEPATHS)
        if os.path.exists(presaved_image_filepaths_path) and not force_reset:
            # Load from previous run
            logging.info(f"Loading list of image file paths from previous run (stored in {presaved_image_filepaths_path})...")
            logging.info("  To run from scratch, pass in force_reset=True to BreastHistopathologyDataset()")
            with open(presaved_image_filepaths_path, "rb") as file:
                image_filepaths = pickle.load(file)
        else:
            # Get all patient IDs, which are numbers (which we match for via regex)
            # Note that each patient has their own subdirectory in `data/`
            p = re.compile("^\d+$")
            patient_ids = [dir for dir in os.listdir(DATA_PATH) if p.match(dir)]
            logging.info(f"Number of patients: {len(patient_ids)}")

            # Iterate through all patient subdirectories and get filenames
            for patient_id in patient_ids:
                patient_dir = os.path.join(DATA_PATH, patient_id)
                for root, dirs, files in os.walk(patient_dir):
                    curr_filepaths = [os.path.join(root, filename) for filename in files]
                    image_filepaths.extend(curr_filepaths)

            # Save for future runs
            with open(presaved_image_filepaths_path, "wb") as file:
                pickle.dump(image_filepaths, file)

        # Extract the label for each image file
        image_labels = [0 for _ in range(len(image_filepaths))]
        for idx, filename in enumerate(image_filepaths):
            # Files are named as <patient_id>_idx..._x..._y..._class<label>.png
            # We extract the label from that filename
            label = int(filename.replace(IMAGE_EXTENSION, "")[-1])
            image_labels[idx] = label

        self.dataframe = pd.DataFrame(
            list(zip(image_filepaths, image_labels)),
            columns=["image_filepath", "label"]
        )

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, idx):

        image_filepath = self.dataframe.loc[idx, "image_filepath"]
        label = self.dataframe.loc[idx, "label"]

        image = Image.open(image_filepath).convert("RGB")
        image_transform = torchvision.transforms.Compose([
            # DINO Vision Transformer expects images of size 224 by 224
            torchvision.transforms.Resize(size=(DINO_IMAGE_SIZE, DINO_IMAGE_SIZE)),
            torchvision.transforms.ToTensor(),
        ])
        image = image_transform(image)

        item = {
            "image_path": image_filepath,
            "image": image,
            "label": label,
        }
        return item

class DinoOriginalBreastHistopathologyDataset(Dataset):
    """
    Dataset used for pretraining Vision Transformers using DINO algorithm
    Note that this dataset is used for the original images when running
    validation during self-supervised learning in pretraining
    """

    def __init__(self, force_reset=False):

        # Get a list of all image files, across all patients
        image_filepaths = list()
        presaved_image_filepaths_path = os.path.join(DATA_PATH, PRESAVED_IMAGE_FILEPATHS)
        if os.path.exists(presaved_image_filepaths_path) and not force_reset:
            # Load from previous run
            logging.info(f"Loading list of image file paths from previous run (stored in {presaved_image_filepaths_path})...")
            logging.info("  To run from scratch, pass in force_reset=True to BreastHistopathologyDataset()")
            with open(presaved_image_filepaths_path, "rb") as file:
                image_filepaths = pickle.load(file)
        else:
            # Get all patient IDs, which are numbers (which we match for via regex)
            # Note that each patient has their own subdirectory in `data/`
            p = re.compile("^\d+$")
            patient_ids = [dir for dir in os.listdir(DATA_PATH) if p.match(dir)]
            logging.info(f"Number of patients: {len(patient_ids)}")

            # Iterate through all patient subdirectories and get filenames
            for patient_id in patient_ids:
                patient_dir = os.path.join(DATA_PATH, patient_id)
                for root, dirs, files in os.walk(patient_dir):
                    curr_filepaths = [os.path.join(root, filename) for filename in files]
                    image_filepaths.extend(curr_filepaths)

            # Save for future runs
            with open(presaved_image_filepaths_path, "wb") as file:
                pickle.dump(image_filepaths, file)

        # Extract the label for each image file
        image_labels = [0 for _ in range(len(image_filepaths))]
        for idx, filename in enumerate(image_filepaths):
            # Files are named as <patient_id>_idx..._x..._y..._class<label>.png
            # We extract the label from that filename
            label = int(filename.replace(IMAGE_EXTENSION, "")[-1])
            image_labels[idx] = label

        self.dataframe = pd.DataFrame(
            list(zip(image_filepaths, image_labels)),
            columns=["image_filepath", "label"]
        )

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, idx):

        image_filepath = self.dataframe.loc[idx, "image_filepath"]
        label = self.dataframe.loc[idx, "label"]

        image = Image.open(image_filepath).convert("RGB")
        image_transform = torchvision.transforms.Compose([
            # DINO Vision Transformer expects images of size 224 by 224
            torchvision.transforms.Resize(size=(PRETRAINING_IMAGE_SIZE, PRETRAINING_IMAGE_SIZE)),
            torchvision.transforms.ToTensor(),
        ])
        image = image_transform(image)

        return image

class DinoPretrainingBreastHistopathologyDataset(Dataset):
    """
    Dataset used for pretraining Vision Transformers using DINO algorithm
    Note that this dataset transforms the images for self-supervised learning

    The DinoOriginalBreastHistopathologyDataset (above) is used for the
    original images when running validation during self-supervised learning
    in pretraining
    """

    def __init__(self, force_reset=False):

        # Get a list of all image files, across all patients
        image_filepaths = list()
        presaved_image_filepaths_path = os.path.join(DATA_PATH, PRESAVED_IMAGE_FILEPATHS)
        if os.path.exists(presaved_image_filepaths_path) and not force_reset:
            # Load from previous run
            logging.info(f"Loading list of image file paths from previous run (stored in {presaved_image_filepaths_path})...")
            logging.info("  To run from scratch, pass in force_reset=True to BreastHistopathologyDataset()")
            with open(presaved_image_filepaths_path, "rb") as file:
                image_filepaths = pickle.load(file)
        else:
            # Get all patient IDs, which are numbers (which we match for via regex)
            # Note that each patient has their own subdirectory in `data/`
            p = re.compile("^\d+$")
            patient_ids = [dir for dir in os.listdir(DATA_PATH) if p.match(dir)]
            logging.info(f"Number of patients: {len(patient_ids)}")

            # Iterate through all patient subdirectories and get filenames
            for patient_id in patient_ids:
                patient_dir = os.path.join(DATA_PATH, patient_id)
                for root, dirs, files in os.walk(patient_dir):
                    curr_filepaths = [os.path.join(root, filename) for filename in files]
                    image_filepaths.extend(curr_filepaths)

            # Save for future runs
            with open(presaved_image_filepaths_path, "wb") as file:
                pickle.dump(image_filepaths, file)

        # Extract the label for each image file
        image_labels = [0 for _ in range(len(image_filepaths))]
        for idx, filename in enumerate(image_filepaths):
            # Files are named as <patient_id>_idx..._x..._y..._class<label>.png
            # We extract the label from that filename
            label = int(filename.replace(IMAGE_EXTENSION, "")[-1])
            image_labels[idx] = label

        self.dataframe = pd.DataFrame(
            list(zip(image_filepaths, image_labels)),
            columns=["image_filepath", "label"]
        )

        self.transformA = torchvision.transforms.RandomResizedCrop(
            (PRETRAINING_IMAGE_SIZE, PRETRAINING_IMAGE_SIZE), scale=(0.5, 1.0))
        self.transformB = torchvision.transforms.RandomResizedCrop(
            (PRETRAINING_IMAGE_SIZE, PRETRAINING_IMAGE_SIZE))

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, idx):

        image_filepath = self.dataframe.loc[idx, "image_filepath"]
        label = self.dataframe.loc[idx, "label"]

        image = Image.open(image_filepath).convert("RGB")
        image_transform = torchvision.transforms.Compose([
            # DINO Vision Transformer expects images of size 224 by 224
            torchvision.transforms.Resize(size=(PRETRAINING_IMAGE_SIZE, PRETRAINING_IMAGE_SIZE)),
            torchvision.transforms.ToTensor(),
        ])
        original_image = image_transform(image)

        # For pretraining
        # Given image is passed through two transforms where one is at leas
        # half as large as the original image
        transformed_imageA = self.transformA(original_image)
        transformed_imageB = self.transformB(original_image)

        return transformed_imageA, transformed_imageB
