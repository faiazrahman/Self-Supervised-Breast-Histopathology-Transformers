"""
python run_training.py --model_type dino
"""

import os
import logging
import argparse

from tqdm import tqdm
import yaml

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloader import BreastHistopathologyDataset, DinoBreastHistopathologyDataset
from model import ResNetIDCDetectionModel, SelfSupervisedDinoIDCDetectionModel, PrintCallback

# Multiprocessing for dataset batching
# NUM_CPUS=40 on Yale Ziva server, NUM_CPUS=24 on Yale Tangra server
# Set to 0 and comment out torch.multiprocessing line if multiprocessing gives errors
NUM_CPUS = 40
# torch.multiprocessing.set_start_method('spawn')

DATA_PATH = "./data"
BATCH_SIZE = 32 # 32 | 64 | 128; 64 maximizes GPU utilization without exceeding CUDA memory
NUM_CLASSES = 2
DEFAULT_GPUS = list(range(torch.cuda.device_count()))

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

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
    parser.add_argument("--model_type", type=str, default=None, help="dino | resnet")
    parser.add_argument("--dino_model_size", type=str, default=None, help="base | small")
    parser.add_argument("--dino_patch_size", type=int, default=None, help="8 | 16")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--dropout_p", type=float, default=None)
    parser.add_argument("--gpus", type=str, help="Comma-separated list of ints with no spaces; e.g. \"0\" or \"0,1\"")
    args = parser.parse_args()

    config = {}
    if args.config is not "":
        with open(str(args.config), "r") as yaml_file:
            config = yaml.safe_load(yaml_file)

    if not args.model_type: args.model_type = config.get("model_type", "dino")
    if not args.dino_model_size: args.dino_model_size = config.get("dino_model_size", "base")
    if not args.dino_patch_size: args.dino_patch_size = config.get("dino_patch_size", 8)
    if not args.batch_size: args.batch_size = config.get("batch_size", 32)
    if not args.learning_rate: args.learning_rate = config.get("learning_rate", 1e-4)
    if not args.num_epochs: args.num_epochs = config.get("num_epochs", 5) # TODO FIXME 10?
    if not args.dropout_p: args.dropout_p = config.get("dropout_p", 0.1)
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE, # TODO args.batch_size,
        num_workers=NUM_CPUS,
        drop_last=True
    )
    logging.info(train_loader)

    hparams = {
        "num_classes": NUM_CLASSES, # TODO args.num_classes
        "learning_rate": args.learning_rate
    }

    model = None
    if args.model_type == "resnet":
        model = ResNetIDCDetectionModel(hparams)
    elif args.model_type == "dino":
        # Add the proper pretrained model checkpoint
        dino_model = "facebook/dino-vit"
        if args.dino_model_size == "small":
            dino_model += "s"
        elif args.dino_model_size == "base":
            dino_model += "b"

        if args.dino_patch_size == 8:
            dino_model += "8"
        elif args.dino_patch_size == 16:
            dino_model += "16"
        hparams["dino_model"] = dino_model

        model = SelfSupervisedDinoIDCDetectionModel(hparams)
    else:
        raise Exception("Given model_type is invalid")

    trainer = None

    latest_checkpoint = ModelCheckpoint(
        filename="latest-{epoch}-{step}",
        monitor="step",
        mode="max",
        every_n_train_steps=100,
        save_top_k=2,
    )
    callbacks = [
        PrintCallback(),
        latest_checkpoint
    ]

    if torch.cuda.is_available():
        # Use all specified GPUs with data parallel strategy
        # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#data-parallel
        trainer = pl.Trainer(
            # gpus=1, # TODO args.gpus,
            # gpus=list(range(torch.cuda.device_count())), # All available GPUs
            gpus=[1,2],
            strategy="dp", # TODO
            callbacks=callbacks,
            enable_checkpointing=True,
            max_epochs=args.num_epochs
        )
    else:
        trainer = pl.Trainer(
            callbacks=callbacks
        )
    logging.info(trainer)

    print(f"Starting training for {args.model_type} model for {args.num_epochs} epochs...")
    trainer.fit(model, train_loader)
