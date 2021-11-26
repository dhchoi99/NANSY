import os
import shutil

from utils.logging.tensorboard import tensorboardLogger


def get_logger(conf):
    return tensorboardLogger(conf)


def save_files(path_save, savefiles):
    os.makedirs(path_save, exist_ok=True)
    for savefile in savefiles:
        path_split = os.path.normpath(savefile).split(os.sep)
        if len(path_split) >= 2:
            dir = os.path.join(path_save, *path_split[:-1])
            os.makedirs(dir, exist_ok=True)
        try:
            shutil.copy2(savefile, os.path.join(path_save, savefile))
        except Exception as e:
            print(f'{e} occured while saving {savefile}')
