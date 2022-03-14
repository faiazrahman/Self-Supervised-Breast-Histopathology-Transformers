import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

print("CUDA available:", torch.cuda.is_available())

class ConvolutionalNeuralNetModel(nn.Module):

    def __init__(self):
        pass

    def forward(self, image, label):
        pass

class SelfSupervisedDinoTransformerModel(nn.Module):

    def __init__(self):
        pass

    def forward(self, image, label):
        pass

class IDCDetectionModelTrainer(pl.LightningModule):

    # TODO: Can we reuse this trainer for all the models?

    def __init__(self, hparams=None):
        pass

    # Required for pl.LightningModule
    def forward(self, image, label):
        # pl.Lightning convention: forward() defines prediction for inference
        return self.model(image, label)

    # Required for pl.LightningModule
    def training_step(self, batch, batch_idx):
        pass

    # Optional for pl.LightningModule
    def training_step_end(self, batch_parts):
        pass

    # Optional for pl.LightningModule
    def test_step(self, batch, batch_idx):
        pass

    # Optional for pl.LightningModule
    def test_epoch_end(self, outputs):
        pass

    # Required for pl.LightningModule
    def configure_optimizers(self):
        pass
