import os
import logging

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transformers import ViTFeatureExtractor, ViTModel

NUM_CLASSES = 2
LEARNING_RATE = 1e-4
DROPOUT_P = 0.1

RESNET_OUT_DIM = 2048
DINO_EMBEDDING_DIM_SMALL = 384 # for dino-vits models
DINO_EMBEDDING_DIM_BASE  = 768 # for dino-vitb models

PATCH_SIZE_S16_MODEL = 'facebook/dino-vits16'
PATCH_SIZE_S8_MODEL = 'facebook/dino-vits8'
PATCH_SIZE_B16_MODEL = 'facebook/dino-vitb16'
PATCH_SIZE_B8_MODEL = 'facebook/dino-vitb8'
DINO_MODEL = PATCH_SIZE_B16_MODEL

losses = []

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

print("CUDA available:", torch.cuda.is_available())

class SelfSupervisedDinoTransformerModel(nn.Module):

    def __init__(
        self,
        num_classes,
        loss_fn,
        hidden_dim=512,
        dropout_p=0.1,
        dino_model=DINO_MODEL,
    ):
        super(SelfSupervisedDinoTransformerModel, self).__init__()
        # NOTE: We run the feature extractor in the pl.LightningModule, i.e.
        # this model receives the image pixel values (from the feature
        # extractor) as input; alternatively, that could be done here by
        # refactoring the pl.LightningModule and uncommenting the following
        # self.feature_extractor = ViTFeatureExtractor.from_pretrained(dino_model)
        self.dino_model = ViTModel.from_pretrained(dino_model)

        dino_embedding_dim = None
        # Note that dino_model is the str name of the model, whereas
        # self.dino_model is the actual ViTModel object
        if dino_model in [PATCH_SIZE_S16_MODEL, PATCH_SIZE_S8_MODEL]:
            dino_embedding_dim = DINO_EMBEDDING_DIM_SMALL
        elif dino_model in [PATCH_SIZE_B16_MODEL, PATCH_SIZE_B8_MODEL]:
            dino_embedding_dim = DINO_EMBEDDING_DIM_BASE

        self.fc1 = torch.nn.Linear(in_features=dino_embedding_dim, out_features=hidden_dim)
        self.fc2 = torch.nn.Linear(in_features=hidden_dim, out_features=num_classes)
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, image_pixel_values, label=None):
        dino_embedding = self.dino_model(pixel_values=image_pixel_values)
        dino_last_hidden_states = dino_embedding.last_hidden_state[:,0]

        hidden = torch.nn.functional.relu(self.fc1(dino_last_hidden_states))
        logits = self.fc2(hidden)

        # nn.CrossEntropyLoss expects raw logits as model output, NOT torch.nn.functional.softmax(logits, dim=1)
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        pred = logits
        loss = self.loss_fn(pred, label)

        return (pred, loss)

class SelfSupervisedDinoIDCDetectionModel(pl.LightningModule):

    def __init__(self, hparams=None):
        super(SelfSupervisedDinoIDCDetectionModel, self).__init__()
        if hparams:
            # Cannot reassign self.hparams in pl.LightningModule; must use update()
            # https://github.com/PyTorchLightning/pytorch-lightning/discussions/7525
            self.hparams.update(hparams)
            logging.info(self.hparams)

        self.learning_rate = self.hparams.get("learning_rate", LEARNING_RATE)

        self.dino_model = self.hparams.get("dino_model", DINO_MODEL)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.dino_model)
        self.model = SelfSupervisedDinoTransformerModel(
            num_classes=self.hparams.get("num_classes", NUM_CLASSES),
            loss_fn=torch.nn.CrossEntropyLoss(),
            dino_model=self.dino_model,
        )
        logging.info(self.dino_model)

        # When reloading the model for evaluation, we will need the
        # hyperparameters as they are now
        self.save_hyperparameters()

    # Required for pl.LightningModule
    def forward(self, image, label):
        # pl.Lightning convention: forward() defines prediction for inference]
        image = torch.tensor(np.stack(self.feature_extractor(image)["pixel_values"]), axis=0)
        return self.model(image, label)

    # Required for pl.LightningModule
    def training_step(self, batch, batch_idx):
        global losses
        # pl.Lightning convention: training_step() defines prediction and
        # accompanying loss for training, independent of forward()
        image, label = batch["image"], batch["label"]

        pred, loss = self.model(image, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        print(loss.item())
        losses.append(loss.item())
        return loss

    # Optional for pl.LightningModule
    def training_step_end(self, batch_parts):
        """
        Aggregates results when training using a strategy that splits data
        from each batch across GPUs (e.g. data parallel)

        Note that training_step returns a loss, thus batch_parts returns a list
        of K loss values (where there are K GPUs being used)
        """
        return sum(batch_parts) / len(batch_parts)

    # Optional for pl.LightningModule
    def test_step(self, batch, batch_idx):
        image, label = batch["image"], batch["label"]
        pred, loss = self.model(image, label)
        pred_label = torch.argmax(pred, dim=1)
        accuracy = torch.sum(pred_label == label).item() / (len(label) * 1.0)
        output = {
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy).cuda()
        }
        print(loss.item(), output['test_acc'])
        return output

    # Optional for pl.LightningModule
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["test_acc"] for x in outputs]).mean()
        logs = {
            'test_loss': avg_loss,
            'test_acc': avg_accuracy
        }

        # pl.LightningModule has some issues displaying the results automatically
        # As a workaround, we can store the result logs as an attribute of the
        # class instance and display them manually at the end of testing
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1088
        self.test_results = logs

        return {
            'avg_test_loss': avg_loss,
            'avg_test_acc': avg_accuracy,
            'log': logs,
            'progress_bar': logs
        }

    # Required for pl.LightningModule
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return optimizer

class ResNetModel(nn.Module):

    def __init__(
        self,
        num_classes,
        loss_fn,
        image_feature_dim,
        hidden_dim=512,
        dropout_p=0.1,
    ):
        super(ResNetModel, self).__init__()
        self.image_encoder = torchvision.models.resnet152(pretrained=True)
        # Overwrite last layer to get features (rather than classification)
        self.image_encoder.fc = torch.nn.Linear(
            in_features=RESNET_OUT_DIM, out_features=image_feature_dim)

        self.fc1 = torch.nn.Linear(in_features=image_feature_dim, out_features=hidden_dim)
        self.fc2 = torch.nn.Linear(in_features=hidden_dim, out_features=num_classes)
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, image, label):
        image_features = torch.nn.functional.relu(self.image_encoder(image))
        hidden = torch.nn.functional.relu(self.fc1(image_features))
        logits = self.fc2(hidden)

        # nn.CrossEntropyLoss expects raw logits as model output, NOT torch.nn.functional.softmax(logits, dim=1)
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        pred = logits
        loss = self.loss_fn(pred, label)

        return (pred, loss)

class ResNetIDCDetectionModel(pl.LightningModule):

    def __init__(self, hparams=None):
        super(ResNetIDCDetectionModel, self).__init__()
        if hparams:
            # Cannot reassign self.hparams in pl.LightningModule; must use update()
            # https://github.com/PyTorchLightning/pytorch-lightning/discussions/7525
            self.hparams.update(hparams)

        self.embedding_dim = self.hparams.get("embedding_dim", 768)
        self.image_feature_dim = self.hparams.get("image_feature_dim", 300)
        self.learning_rate = self.hparams.get("learning_rate", LEARNING_RATE)

        self.model = ResNetModel(
            num_classes=self.hparams.get("num_classes", NUM_CLASSES),
            loss_fn=torch.nn.CrossEntropyLoss(),
            image_feature_dim=self.image_feature_dim,
            dropout_p=self.hparams.get("dropout_p", DROPOUT_P)
        )

    # Required for pl.LightningModule
    def forward(self, image, label):
        # pl.Lightning convention: forward() defines prediction for inference
        return self.model(image, label)

    # Required for pl.LightningModule
    def training_step(self, batch, batch_idx):
        global losses
        # pl.Lightning convention: training_step() defines prediction and
        # accompanying loss for training, independent of forward()
        image, label = batch["image"], batch["label"]

        pred, loss = self.model(image, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        print(loss.item())
        losses.append(loss.item())
        return loss

    # Optional for pl.LightningModule
    def training_step_end(self, batch_parts):
        """
        Aggregates results when training using a strategy that splits data
        from each batch across GPUs (e.g. data parallel)

        Note that training_step returns a loss, thus batch_parts returns a list
        of K loss values (where there are K GPUs being used)
        """
        return sum(batch_parts) / len(batch_parts)

    # Optional for pl.LightningModule
    def test_step(self, batch, batch_idx):
        image, label = batch["image"], batch["label"]
        pred, loss = self.model(image, label)
        pred_label = torch.argmax(pred, dim=1)
        accuracy = torch.sum(pred_label == label).item() / (len(label) * 1.0)
        output = {
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy).cuda()
        }
        print(loss.item(), output['test_acc'])
        return output

    # Optional for pl.LightningModule
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["test_acc"] for x in outputs]).mean()
        logs = {
            'test_loss': avg_loss,
            'test_acc': avg_accuracy
        }

        # pl.LightningModule has some issues displaying the results automatically
        # As a workaround, we can store the result logs as an attribute of the
        # class instance and display them manually at the end of testing
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1088
        self.test_results = logs

        return {
            'avg_test_loss': avg_loss,
            'avg_test_acc': avg_accuracy,
            'log': logs,
            'progress_bar': logs
        }

    # Required for pl.LightningModule
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return optimizer

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training started...")

    def on_train_end(self, trainer, pl_module):
        print("Training done...")
        global losses
        for loss_val in losses:
            print(loss_val)
