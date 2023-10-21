import pytorch_lightning as pl
from models.gan import GAN
from models.gan_nearest_upsampling import GANNearest
from datamodules.idd_datamodule import IDDDataModule
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def main():

    model = GANNearest()

    idd_datamodule =  IDDDataModule('/data/divy/idd20k_lite/full_dataset/small_idd20kII/leftImg8bit/train/*/*.jpg',
                                    '../idd20k_lite/leftImg8bit/val/*/*_image.jpg',
                                    batch_size=64, num_workers=12)
    idd_datamodule.setup()
    train_dl = idd_datamodule.train_dataloader()
    valid_dl = idd_datamodule.val_dataloader()

    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="total_g_loss", mode='min', 
                                          filename="{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}")
    logger = TensorBoardLogger("runs/lightning_logs", name="IDD")

    trainer = pl.Trainer(strategy='ddp_find_unused_parameters_true', max_epochs=120, accelerator='gpu',
                          devices=[0], default_root_dir='runs', callbacks=[checkpoint_callback],
                          logger=logger, log_every_n_steps=54)
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)

if __name__ == '__main__':
    main()

