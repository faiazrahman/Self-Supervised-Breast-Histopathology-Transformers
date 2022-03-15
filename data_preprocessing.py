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
    # Calling the BreastHistopathologyDataset constructor (i.e. __init__) will
    # run all the necessary data preprocessing steps and dump any slow-to-load
    # data into serialized .pkl files; on subsequent runs, initializing new
    # BreastHistopathologyDataset objects will be much faster
    full_dataset = BreastHistopathologyDataset()
    logging.info("Total dataset size: {}".format(len(full_dataset)))
    logging.info(full_dataset)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(6)
    )
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(train_dataset)
    logging.info(f"Test dataset size: {len(test_dataset)}")
    logging.info(test_dataset)

    # TODO: rm, this is just making sure the subsets are still iterable
    # train_count = 0
    # for batch_idx, batch_items in enumerate(train_dataset):
    #     # print(batch_items)
    #     # print(batch_items["image"])
    #     train_count += 1
    #     if train_count > 500:
    #         break

    # test_count = 0
    # for batch_idx, batch_items in enumerate(test_dataset):
    #     test_count += 1
    #     if test_count > 500:
    #         break
    # print(train_count, test_count)