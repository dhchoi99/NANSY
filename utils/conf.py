from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


def _load(conf: DictConfig):
    if 'load' not in conf.keys():
        return conf

    if isinstance(conf['load'], str):
        load_path = conf['load']
        try:
            conf['load'] = OmegaConf.load(load_path)
            conf['load']['load'] = load_path
        except Exception as e:
            print(f'Loading {load_path} as omegaconf DictConfig failed')
            raise e

    if isinstance(conf['load'], DictConfig) or isinstance(conf['load'], dict):
        for key, value in conf['load'].items():
            if key == 'load':
                continue

            if key in conf.keys():
                print(f'Warning: key {key} exists, overwriting')

            try:
                conf[key] = OmegaConf.load(value)
            except Exception as e:
                print(f'Loading {value} as omegaconf DictConfig failed')
                raise e
    else:
        raise NotImplementedError

    return conf


def set_conf(conf):
    if isinstance(conf, str):
        conf = OmegaConf.load(conf)
    conf = _load(conf)
    return conf
