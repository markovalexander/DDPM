import pytorch_lightning as pl
import torch

from torch import nn, optim

from lib.model import UNet
from lib.diffusion import GaussianDiffusion, make_beta_schedule
from lib.samplers import get_time_sampler
from utils import accumulate


class DDP(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()

        self.conf = conf
        self.save_hyperparameters(conf)
        self.shape = (3, self.conf.data.dataset.resolution, self.conf.data.dataset.resolution)

        predict_var = (self.conf.model.diffusion.model_var_type.find("learned") >= 0)
        self.model = UNet(**self.conf.model.unet, predict_var=predict_var)
        self.ema = UNet(**self.conf.model.unet, predict_var=predict_var)
        self.betas = make_beta_schedule(**self.conf.model.schedule)
        self.diffusion = GaussianDiffusion(betas=self.betas, **self.conf.model.diffusion)
        self.sampler = get_time_sampler(self.conf.model.sampler)(self.diffusion)

    def forward(self, x):
        return self.diffusion.p_sample_loop(self.model, x.shape)

    def configure_optimizers(self):
        if self.conf.model.optimizer.type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.conf.model.optimizer.lr)
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_ix):
        img, _ = batch
        # time = torch.randint(size=(img.shape[0],), low=0, high=self.conf.model.schedule.n_timestep,
        #                      dtype=torch.int64, device=img.device)
        time, weights = self.sampler.sample(img.size(0), device=img.device)
        loss = self.diffusion.training_losses(self.model, img, time)
        self.sampler.update_with_all_losses(time, loss)
        loss = torch.mean(loss * weights)
        accumulate(self.ema, self.model.module if isinstance(self.model, nn.DataParallel) else self.model, 0.9999)

        self.log('train_loss', loss, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_ix):
        img, _ = batch
        time, weights = self.sampler.sample(img.size(0), device=img.device)
        loss = self.diffusion.training_losses(self.model, img, time)
        loss = torch.mean(loss * weights)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        self.log('val_loss', avg_loss, logger=True)
        return {'val_loss': avg_loss}
