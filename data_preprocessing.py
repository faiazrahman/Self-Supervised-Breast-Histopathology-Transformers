import os
import re
import logging
import pickle

import torch

from dataloader import BreastHistopathologyDataset

DATA_PATH = "./data"
IMAGE_NAMES_FILENAME = "patient_ids.pkl"

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

if __name__ == "__main__":
    # # Get all patient IDs, which are numbers (which we match for via regex)
    # # Note that each patient has their own subdirectory in `data/`
    # p = re.compile("^\d+$")
    # patient_ids = [dir for dir in os.listdir(DATA_PATH) if p.match(dir)]
    # print(f"Number of patients: {len(patient_ids)}")

    # # Get a list of all image files, across all patients
    # image_files = list()
    # presaved_image_names_path = os.path.join(DATA_PATH, IMAGE_NAMES_FILENAME)
    # if os.path.exists(presaved_image_names_path):
    #     # Load from previous run
    #     with open(presaved_image_names_path, "rb") as file:
    #         image_files = pickle.load(file)
    # else:
    #     # Iterate through all patient subdirectories and get filenames
    #     for patient_id in patient_ids:
    #         patient_dir = os.path.join(DATA_PATH, patient_id)
    #         for root, dirs, files in os.walk(patient_dir):
    #             image_files.extend(files)

    #     # Save for future runs
    #     with open(presaved_image_names_path, "wb") as file:
    #         pickle.dump(image_files, file)

    # print(len(image_files))

    full_dataset = BreastHistopathologyDataset()
    logging.info("Total dataset size: {}".format(len(full_dataset)))
    logging.info(full_dataset)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(6)
    )
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    logging.info(train_dataset)
    logging.info("Test dataset size: {}".format(len(test_dataset)))
    logging.info(test_dataset)

    # TODO: rm, this is just making sure the subsets are still iterable
    train_count = 0
    for batch_idx, batch_items in enumerate(train_dataset):
        print(batch_items)
        print(batch_items["image"])
        train_count += 1
        break

    # test_count = 0
    # for batch_idx, batch_items in enumerate(test_dataset):
    #     test_count += 1
    # print(train_count, test_count)