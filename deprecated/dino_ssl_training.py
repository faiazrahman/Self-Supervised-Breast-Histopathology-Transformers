from pathlib import Path
from ctypes import resize
from tkinter.tix import IMAGE
import copy
import logging
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from deepspeed.ops.adam import FusedAdam

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

IMAGE_SIZE = 256
PATCH_SIZE = 16 # 8 | 16
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

class InHouseDinoTransformerModel(nn.Module):
    """
    Vision Transformer for pretraining with DINO algorithm, which can then be
    fitted with a classifier head for IDC classification
    """
    def __init__(self, d_model, n_head, num_layers):
        super().__init__()

        # Transformer
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head)
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=num_layers)

        # Positional embeddings
        w_pos = torch.randn(NUM_PATCHES, d_model) / d_model ** 0.5
        cls_token = torch.randn(1, d_model) / d_model ** 0.5
        self.register_parameter("pos_embed", nn.Parameter(w_pos))
        self.register_parameter("cls_token", nn.Parameter(cls_token))

        # Pixel projection
        self.linear = nn.Linear(2 * d_model, d_model)
        self.norm1 = nn.LayerNorm(2 * d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size = len(x)
        position = torch.stack([self.pos_embed] * batch_size)
        x = torch.cat([x, position], dim=-1)
        pixel_proj = self.norm2(F.relu(self.linear(self.norm1(x))))
        batched_cls_token = torch.stack([self.cls_token]*batch_size)
        cls_x = torch.cat([batched_cls_token, pixel_proj], dim=1)

        cls_x.transpose_(0, 1)
        return F.normalize(self.encoder(cls_x)[0, ...], dim=-1)

class HLoss:
    def __init__(self, temperature_t, temperature_s):
        self.temperature_t = temperature_t
        self.temperature_s = temperature_s

    def __call__(self, t, s, center):
        t = F.softmax((t.detach() - center) / self.temperature_t, dim=1)
        log_s = F.log_softmax(s / self.temperature_s, dim=1)
        return -(t * log_s).sum(dim=1).mean()

class InHouseDinoPretrainingModel(pl.LightningModule):
    """ PyTorch Lightning module for pretraining ViT with DINO """

    def __init__(self, teacher, lr, loss_fn, dim, center_momentum,
        param_momentum):
        super().__init__()
        self.teacher = teacher
        self.student = copy.deepcopy(teacher)
        self.lr = lr
        self.loss_fn = loss_fn
        self.c_mom = center_momentum
        self.p_mom = param_momentum

        # Buffers are parameters of the model that should be saved and restored
        # but not trained by the optimizer
        # Buffers won't be returned in model.parameters() so the optimizer
        # won't have a change to update them
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
        self.register_buffer("center", torch.zeros((1, dim)))

        for p in self.teacher.parameters():
            p.requires_grad = False

    def loss_calculation(self, batch):
        x1, x2 = batch
        s1, s2 = self.student(x1), self.student(x2)
        t1, t2 = self.teacher(x1), self.teacher(x2)

        loss = self.loss_fn(t1, s2, self.center) + self.loss_fn(t2, s1, self.center)
        empirical_center = F.normalize(
            torch.cat([t1, t2]).mean(dim=0, keepdims=True), dim=-1)
        return loss, empirical_center

    def training_step(self, batch, *args):
        loss, empirical_center = self.loss_calculation(batch)
        self.log(name="train_loss", value=loss, on_step=True, on_epoch=True)
        self.center = F.normalize(self.c_mom * self.center + (1 - self.c_mom) * empirical_center, dim=-1)
        for s_p, t_p in zip(self.student.parameters(), self.teacher.parameters()):
            t_p.data = self.p_mom * t_p.data + (1 - self.p_mom) * s_p.data
        return loss

    def validation_step(self, images, *args):
        return self.teacher(images)

    def configure_optimizers(self):
        return FusedAdam(self.student.parameters(), lr=self.lr)

if __name__ == "__main__":
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
        batch_size=BATCH_SIZE,
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
        batch_size=BATCH_SIZE,
        num_workers=NUM_CPUS,
        drop_last=True,
        collate_fn=CollateSingleImage()
    )
    logging.info(val_loader)

    teacher = InHouseDinoTransformerModel(NUM_PIXELS, TRANSFORMER_N_HEADS, TRANSFORMER_N_LAYERS)
    h_loss = HLoss(TEACHER_TEMPERATURE, STUDENT_TEMPERATURE)
    model = InHouseDinoPretrainingModel(teacher, LEARNING_RATE, h_loss, NUM_PIXELS, CENTER_MOMENTUM, TEACHER_MOMENTUM)

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        gpus=[5,7],
        strategy="dp",
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_loader, val_loader)
