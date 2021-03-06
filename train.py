import argparse
import glob
import os

from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from trainer import Trainer
import utils.logging


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/train_nansy.yaml")

    parser.add_argument('-g', '--gpus', type=str,
                        help="")
    parser.add_argument('-p', '--resume_checkpoint_path', type=str, default=None,
                        help="path of checkpoint for resuming")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    conf = OmegaConf.load(args.config)

    conf.logging.log_dir = os.path.join(conf.logging.log_dir, str(conf.logging.seed))
    os.makedirs(conf.logging.log_dir, exist_ok=True)

    save_file_dir = os.path.join(conf.logging.log_dir, 'code')
    os.makedirs(save_file_dir, exist_ok=True)
    savefiles = []
    for reg in conf.logging.save_files:
        savefiles += glob.glob(reg)
    utils.logging.save_files(save_file_dir, savefiles)

    checkpoint_dir = os.path.join(conf.logging.log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, **conf.pl.checkpoint.callback)

    tensorboard_dir = os.path.join(conf.logging.log_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    logger = TensorBoardLogger(tensorboard_dir)
    logger.log_hyperparams(conf)

    trainer = pl.Trainer(
        logger=logger,
        gpus=args.gpus,
        callbacks=[checkpoint_callback],
        weights_save_path=checkpoint_dir,
        resume_from_checkpoint=args.resume_checkpoint_path,
        **conf.pl.trainer
    )

    model = Trainer(conf)  # TODO get trainer from conf
    trainer.fit(model)


if __name__ == '__main__':
    main()
