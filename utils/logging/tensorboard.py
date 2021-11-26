import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class tensorboardLogger(SummaryWriter):
    def __init__(self, conf):
        self.conf = conf
        super(tensorboardLogger, self).__init__(log_dir=self.conf.log_dir)

    def write_loss(self, loss, mode, step):
        for key, value in loss.items():
            if isinstance(value, torch.Tensor):
                self.add_scalar(tag=f'{mode}_loss/{key}', scalar_value=value, global_step=step)

    def write_log(self, log, mode, step):
        imgs = []
        img_keys = []

        for key, value in log.items():
            value = value.detach().cpu()
            if 'img' in key:
                if value.ndim == 3:  # B x H x W
                    imgs.append(torch.repeat_interleave(value[0].unsqueeze(0), 3, dim=0))
                    img_keys.append(key)
                elif value.ndim == 4:  # B x C x H x W
                    imgs.append(value[0])
                    img_keys.append(key)

        if len(img_keys) > 0:
            imgs = torch.cat(imgs, dim=-1)
            img_key = ' | '.join(img_keys)
            self.add_image(tag=f'{mode}/{img_key}', img_tensor=imgs, global_step=step)
