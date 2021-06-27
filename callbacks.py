import pytorch_lightning as pl
import torch
import wandb

from torchvision.utils import make_grid
from utils import progressive_samples_fn


def get_checkpoint_callback(path):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=path,
                                                       filename='ddp_{epoch:02d}-{val_loss:.2f}',
                                                       monitor='val_loss',
                                                       verbose=True,
                                                       save_last=True,
                                                       save_weights_only=True,
                                                       every_n_val_epochs=5)
    return checkpoint_callback


class ImageLoggerCallback(pl.callbacks.Callback):
    def __init__(self, n_samples=16):
        super().__init__()
        self.n_samples = n_samples

    def on_validation_end(self, trainer, pl_module):
        shape = (self.n_samples, *pl_module.shape)
        sample = progressive_samples_fn(pl_module.ema, pl_module.diffusion, shape,
                                        device='cuda' if pl_module.on_gpu else 'cpu')

        samples_grid = make_grid(sample['samples'], nrow=self.n_samples // 4)
        samples_log = wandb.Image(samples_grid.detach().cpu().numpy().transpose((1, 2, 0)),
                                  caption='generated samples')

        progressive_grid = make_grid(sample['progressive_samples'].reshape(-1, *pl_module.shape))
        progressive_log = wandb.Image(progressive_grid.detach().cpu().numpy().transpose((1, 2, 0)),
                                      caption='progressive samples')

        wandb.log({"samples": samples_log, 'progressive_samples': progressive_log})
