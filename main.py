import warnings

import torch
import torch.optim as optim

import pytorch_lightning as pl

from PIL import ImageFile
from argparse import ArgumentParser

from torch.utils.data import Dataset

from modules import NT_Xent, TransformsSimCLR, SimCLR, ImageDataset


warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SimCLRModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.model = SimCLR(hparams.projection_dim)
        self.criterion = NT_Xent(hparams.batch_size, hparams.temperature)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x_i, x_j = batch

        h_i, z_i = self.model(x_i)
        h_j, z_j = self.model(x_j)

        loss = self.criterion(z_i, z_j)
        log = {'train_loss': loss}
        
        return {'loss': loss, 'log': log}
    
    def validation_step(self, batch, batch_idx):
        x_i, x_j = batch

        h_i, z_i = self.model(x_i)
        h_j, z_j = self.model(x_j)

        loss = self.criterion(z_i, z_j)
        
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': log}
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3)
    
    def train_dataloader(self):
        dataset = ImageDataset(
            self.hparams.train_path,
            transform=TransformsSimCLR(self.hparams.img_size)
        )
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            drop_last=True,
            shuffle=True
        )
        return train_loader
    
    def val_dataloader(self):
        dataset = ImageDataset(
            self.hparams.valid_path,
            transform=TransformsSimCLR(self.hparams.img_size)
        )
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            drop_last=True,
            shuffle=False
        )
        return train_loader


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--projection_dim', type=int, default=256)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--train_path', type=str, default='./data/train')
    parser.add_argument('--valid_path', type=str, default='./data/valid')

    args = parser.parse_args()

    model = SimCLRModel(args)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        deterministic=True
    )
    trainer.fit(model)
