import requests
import logging
from PIL import Image

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import ViTFeatureExtractor, ViTModel

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloader import BreastHistopathologyDataset, DinoBreastHistopathologyDataset
from model import ResNetModel, IDCDetectionModel, PrintCallback

NUM_CLASSES = 2
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_CPUS = 0 # 0 | 40

DINO_EMBEDDING_DIM = 384

class SelfSupervisedDinoTransformerModel(nn.Module):

    def __init__(
        self,
        num_classes,
        loss_fn,
        # image_feature_dim,
        dino_embedding_dim,
        hidden_dim=512,
        dropout_p=0.1,
    ):
        super(SelfSupervisedDinoTransformerModel, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')
        self.dino_model = ViTModel.from_pretrained('facebook/dino-vits16')

        self.fc1 = torch.nn.Linear(in_features=dino_embedding_dim, out_features=hidden_dim)
        self.fc2 = torch.nn.Linear(in_features=hidden_dim, out_features=num_classes)
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)

    # def forward(self, image, label):
    #     # FIXME THIS LINE RAISED THE FOLLOWING ERROR ---
    #     #  TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    #     image = image.cpu()
    #     inputs = self.feature_extractor(images=image, return_tensors="pt")
    #     dino_embedding = self.dino_model(**inputs)
    #     dino_last_hidden_states = dino_embedding.last_hidden_state
    #     print(list(dino_last_hidden_states.shape))

    #     hidden = torch.nn.functional.relu(self.fc1(dino_last_hidden_states))
    #     logits = self.fc2(hidden)

    #     # nn.CrossEntropyLoss expects raw logits as model output, NOT torch.nn.functional.softmax(logits, dim=1)
    #     # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    #     pred = logits
    #     loss = self.loss_fn(pred, label)

    #     return (pred, loss)

    def forward(self, image_pixel_values, label=None):
        dino_embedding = self.dino_model(pixel_values=image_pixel_values)
        dino_last_hidden_states = dino_embedding.last_hidden_state

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

        self.learning_rate = self.hparams.get("learning_rate", LEARNING_RATE)

        self.model = SelfSupervisedDinoTransformerModel(
            num_classes=self.hparams.get("num_classes", NUM_CLASSES),
            loss_fn=torch.nn.CrossEntropyLoss(),
            dino_embedding_dim=DINO_EMBEDDING_DIM,
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

if __name__ == "__main__":
    # # Facebook AI
    # vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    # print(vits16)

    # # Hugging Face
    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # image = Image.open(requests.get(url, stream=True).raw)

    # feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')
    # model = ViTModel.from_pretrained('facebook/dino-vits16')
    # inputs = feature_extractor(images=image, return_tensors="pt")
    # outputs = model(**inputs)
    # last_hidden_states = outputs.last_hidden_state
    # print(list(last_hidden_states.shape))

    full_dataset = DinoBreastHistopathologyDataset()
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

    # example = train_dataset[0]
    # example_image, example_label = example["image"], example["label"]
    # example_label = np.array([example_label])
    # example_label = torch.from_numpy(example_label)

    # out = model(example_image, example_label)
    # print(out)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE, # TODO args.batch_size,
        num_workers=NUM_CPUS,
        drop_last=True
    )
    logging.info(train_loader)

    hparams = {
        "num_classes": NUM_CLASSES, # TODO args.num_classes
        "learning_rate": LEARNING_RATE,
    }

    # model = SelfSupervisedDinoTransformerModel(
    #     num_classes=2,
    #     loss_fn=torch.nn.CrossEntropyLoss(),
    #     dino_embedding_dim=384,
    # )
    model = SelfSupervisedDinoIDCDetectionModel(hparams)
    print(model)

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
            gpus=list(range(torch.cuda.device_count())), # All available GPUs
            strategy="dp", # TODO
            callbacks=callbacks,
            enable_checkpointing=True
        )
    else:
        trainer = pl.Trainer(
            callbacks=callbacks
        )
    logging.info(trainer)

    trainer.fit(model, train_loader)
