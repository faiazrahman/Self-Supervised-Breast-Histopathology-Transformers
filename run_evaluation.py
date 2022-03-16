import sys
import os
from pathlib import Path
import logging
import argparse
import enum

import pandas as pd
import numpy as np

from tqdm import tqdm
import yaml

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision

import pytorch_lightning as pl

from dataloader import BreastHistopathologyDataset, DinoBreastHistopathologyDataset
from model import ResNetIDCDetectionModel, SelfSupervisedDinoResNetIDCDetectionModel, PrintCallback

# Multiprocessing for dataset batching
# NUM_CPUS=40 on Yale Ziva server, NUM_CPUS=24 on Yale Tangra server
# Set to 0 and comment out torch.multiprocessing line if multiprocessing gives errors
NUM_CPUS = 40
# torch.multiprocessing.set_start_method('spawn')

DATA_PATH = "./data"
PL_ASSETS_PATH = "./lightning_logs"
BATCH_SIZE = 32
NUM_CLASSES = 2
DEFAULT_GPUS = list(range(torch.cuda.device_count()))

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

def get_checkpoint_filename_from_dir(path):
    return os.listdir(path)[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="config.yaml file with experiment configuration")

    # We default all hyperparameters to None so that their default values can
    # be taken from a config file; if the config file is not specified, then we
    # use the given default values in the `config.get()` calls (see below)
    # Thus the order of precedence for hyperparameter values is
    #   passed manually as an arg -> specified in given config file -> default
    # This allows experiments defined in config files to be easily replicated
    # while tuning specific parameters via command-line args
    parser.add_argument("--trained_model_version", type=int, default=None, help="Version number (int) of trained model checkpoints, as stored in lightning_logs/")
    parser.add_argument("--model_type", type=str, default=None, help="dino | resnet; Must match the type of the trained model being evaluated")
    parser.add_argument("--gpus", type=str, help="Comma-separated list of ints with no spaces; e.g. \"0\" or \"0,1\"")
    args = parser.parse_args()

    config = {}
    if args.config is not "":
        with open(str(args.config), "r") as yaml_file:
            config = yaml.safe_load(yaml_file)

    if not args.model_type:
        args.model_type = config.get("model_type", "resnet")
    if args.gpus:
        args.gpus = [int(gpu_num) for gpu_num in args.gpus.split(",")]
    else:
        args.gpus = config.get("gpus", DEFAULT_GPUS)

    full_dataset = None
    if args.model_type == "resnet":
        full_dataset = BreastHistopathologyDataset()
    elif args.model_type == "dino":
        full_dataset = DinoBreastHistopathologyDataset()
    else:
        raise Exception("Given model_type is invalid")

    # full_dataset = BreastHistopathologyDataset() # TODO rm
    logging.info("Total dataset size: {}".format(len(full_dataset)))
    logging.info(full_dataset)

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
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(train_dataset)
    logging.info(f"Test dataset size: {len(test_dataset)}")
    logging.info(test_dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE, # TODO args.batch_size,
        num_workers=NUM_CPUS,
        drop_last=True
    )
    logging.info(test_loader)

    hparams = {
        "num_classes": NUM_CLASSES, # TODO args.num_classes
    }

    checkpoint_path = None
    if args.trained_model_version:
        assets_version = None
        if isinstance(args.trained_model_version, int):
            assets_version = "version_" + str(args.trained_model_version)
        elif isinstance(args.trained_model_version, str):
            assets_version = args.trained_model_version
        else:
            raise Exception("assets_version must be either an int (i.e. the version number, e.g. 16) or a str (e.g. \"version_16\"")
        checkpoint_path = os.path.join(PL_ASSETS_PATH, assets_version, "checkpoints")
    else:
        raise Exception("A trained model must be specified for evaluation, by version number (in default PyTorch Lightning assets path ./lightning_logs)")

    checkpoint_filename = get_checkpoint_filename_from_dir(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, checkpoint_filename)
    logging.info(checkpoint_path)

    model = None
    if args.model_type == "resnet":
        model = ResNetIDCDetectionModel.load_from_checkpoint(checkpoint_path)
    elif args.model_type == "dino":
        model = SelfSupervisedDinoResNetIDCDetectionModel.load_from_checkpoint(checkpoint_path)
    else:
        raise Exception("Given model_type is invalid")

    trainer = None
    if torch.cuda.is_available():
        # Use all specified GPUs with data parallel strategy
        # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#data-parallel
        callbacks = [PrintCallback()]
        trainer = pl.Trainer(
            gpus=args.gpus,
            strategy="dp",
            callbacks=callbacks,
        )
    else:
        trainer = pl.Trainer()
    logging.info(trainer)

    trainer.test(model, dataloaders=test_loader)
    # pl.LightningModule has some issues displaying the results automatically
    # As a workaround, we can store the result logs as an attribute of the
    # class instance and display them manually at the end of testing
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/1088
    results = model.test_results

    print(checkpoint_path)
    print(results)
    logging.info(checkpoint_path)
    logging.info(results)
