import hydra
import pytorch_lightning as pl
import logging
import wandb

from lib.datasets import get_datamodule
from ddp import DDP

from callbacks import get_checkpoint_callback, ImageLoggerCallback
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

logger = logging.getLogger(__name__)


@hydra.main(config_path='config', config_name="config")
def train_or_generate(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    if cfg.task.train:
        train(cfg)
    else:
        pass


def train(cfg):
    run = wandb.init(project="denoising_diffusion", config=cfg, group="train")
    name = str(run.name)
    pl_logger = pl.loggers.WandbLogger()

    checkpoint_path = hydra.utils.get_original_cwd() / Path(cfg.task.trainer.ckpt_dir) / Path(name)
    pl.seed_everything(cfg.task.seed)
    
    logger.info("checkpoints will be saved in " + str(checkpoint_path))
    denoising_diffusion_model = DDP(cfg.task)
    dm = get_datamodule(cfg.task.data)
    checkpoint_callback = get_checkpoint_callback(checkpoint_path)
    image_logger_callback = ImageLoggerCallback(16)

    trainer = pl.Trainer(gpus=cfg.task.trainer.n_gpu,
                         max_steps=cfg.task.trainer.n_iter,
                         precision=cfg.task.trainer.precision,
                         gradient_clip_val=1.,
                         progress_bar_refresh_rate=20,
                         checkpoint_callback=True,
                         callbacks=[checkpoint_callback, image_logger_callback],
                         logger=pl_logger,
                         accelerator='dp')
    trainer.fit(denoising_diffusion_model, dm)


if __name__ == "__main__":
    train_or_generate()
