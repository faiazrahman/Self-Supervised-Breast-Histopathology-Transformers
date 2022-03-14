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

# from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

DATA_PATH = "./data"
PL_ASSETS_PATH = "./lightning_logs"
IMAGE_EXTENSION = ".png"
PRESAVED_IMAGE_FILENAMES = "image_filenames.pkl"

class BreastHistopathologyDataset(Dataset):

    def __init__(self, force_reset=False):
        # Get a list of all image files, across all patients
        image_files = list()
        presaved_image_names_path = os.path.join(DATA_PATH, PRESAVED_IMAGE_FILENAMES)
        if os.path.exists(presaved_image_names_path) and not force_reset:
            # Load from previous run
            with open(presaved_image_names_path, "rb") as file:
                image_files = pickle.load(file)
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
                    image_files.extend(files)

            # Save for future runs
            with open(presaved_image_names_path, "wb") as file:
                pickle.dump(image_files, file)

        image_labels = [0 for _ in range(len(image_files))]
        for idx, filename in enumerate(image_files):
            # Files are named as <patient_id>_idx..._x..._y..._class<label>.png
            # We extract the label from that filename
            label = int(filename.replace(IMAGE_EXTENSION, "")[-1])
            image_labels[idx] = label

        self.dataframe = pd.DataFrame(
            list(zip(image_files, image_labels)),
            columns=["image_filename", "label"]
        )

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, idx):

        item_id = self.dataframe.loc[idx, "image_filename"]
        label = self.dataframe.loc[idx, "label"]

        # FIXME
        image = None

        item = {
            "id": item_id,
            "image": image,
            "label": label,
        }
        return item
