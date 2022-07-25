# NANSY:

Unofficial Pytorch Implementation of
[Neural Analysis and Synthesis: Reconstructing Speech from Self-Supervised Representations](https://arxiv.org/pdf/2110.14513.pdf)

## Notice

### Papers' Demo

Check Authors'
[Demo page](https://harsh-grenadilla-e40.notion.site/Demo-page-for-NANSY-37d4fd8ffb514765a2b234b04c8fc0f6)

### Sample-Only Demo Page

Check [Demo Page](https://dhchoi99.github.io/NANSY)

### Concerns

```
Among the various controllabilities, it is rather obvious that the voice conversion technique can be misused and potentially harm other people.
More concretely, there are possible scenarios where it is being used by random unidentified users and contributing to spreading fake news.
In addition, it can raise concerns about biometric security systems based on speech.
To mitigate such issues, the proposed system should not be released without a consent so that it cannot be easily used by random users with malicious intentions.
That being said, there is still a potential for this technology to be used by unidentified users.
As a more solid solution, therefore, we believe a detection system that can discriminate between fake and real speech should be developed.
```

We provide both pretrained checkpoint of Discriminator network and inference code for this concern.

## Environment

### Requirements

`pip install -r requirements.txt`

### Docker

#### Image

If using cu113 compatible environment, use [Dockerfile](./Dockerfile)  
If using cu102 compatible environment, use [Dockerfile-cu102](./Dockerfile_cu102)

`docker build -f Dockerfile -t nansy:v0.0 .`

#### Container

After building appropriate image, use docker-compose or docker to run a container.  
You may want to modify `docker-compose.yml` or `docker_run_script.sh`

`docker-compose -f docker-compose.yml run --service-ports --name CONTAINER_NAME nansy_container bash`  
or  
[`bash docker_run_script.sh`](./docker_run_script.sh)

### Pretrained hifi-gan

Download pretrained hifi-gan config and checkpoint  
from [hifi-gan](https://github.com/jik876/hifi-gan)
to `./configs/hifi-gan/UNIVERSAL_V1`

### Pretrained Checkpoints

TODO

## Datasets

Datasets used when training are:

- VCTK:
  - CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)
  - https://datashare.ed.ac.uk/handle/10283/3443
    For `data/VCTK-Corpus/vctk_22k_train.txt`, `vctk_22k_val.txt` and `vctk_22k_test.txt`, use files at [mindslab-ai/cotatron](https://github.com/mindslab-ai/cotatron/tree/master/datasets/metadata).
- LibriTTS:
  - Large-scale corpus of English speech derived from the original materials of the LibriSpeech corpus
  - https://openslr.org/60/
  - train-clean-360 set
    For `data/LibriTTS/libritts_train_clean_360_audiopath_text_sid_train.txt` and `libritts_train_clean_360_audiopath_text_sid_val.txt`, use files at [mindslab-ai/univnet](https://github.com/mindslab-ai/univnet/tree/master/datasets/metadata).
- CSS10:
  - CSS10: A Collection of Single Speaker Speech Datasets for 10 Languages
  - https://github.com/Kyubyong/css10

Place datasets at `data`

### Custom Datasets

Write your own code!  
If inheriting `datasets.custom.CustomDataset`, `self.data` should be as:

```
self.data: list
self.data[i]: dict must have:
    'wav_path_22k': str = path_to_22k_wav_file
    'wav_path_16k': str = (optional) path_to_16k_wav_file
    'speaker_id': str = speaker_id
```

## Train

If you prefer `pytorch-lightning`,
`python train.py -g 1`

```python
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/train_nansy.yaml")
parser.add_argument('-g', '--gpus', type=str,
                    help="number of gpus to use")
parser.add_argument('-p', '--resume_checkpoint_path', type=str, default=None,
                    help="path of checkpoint for resuming")
args = parser.parse_args()
return args
```

else `python train_torch.py` # TODO, not completely supported now

### Configs Description

Edit `configs/train_nansy.yaml`.

#### Dataset settings

- Adjust `datasets.*.datasets` list.
  - Paths to dataset config files should be in the list

```yaml
datasets:
  train:
    class: datasets.base.MultiDataset
    datasets:
      ["configs/datasets/vctk.yaml", "configs/datasets/libritts360.yaml"]

    mode: train
    batch_size: 32 # Depends on GPU Memory, Original paper used 32
    shuffle: True
    num_workers: 16 # Depends on available CPU cores

  eval:
    class: datasets.base.MultiDataset
    datasets:
      ["configs/datasets/vctk.yaml", "configs/datasets/libritts360.yaml"]

    mode: eval
    batch_size: 32
    shuffle: False
    num_workers: 4
```

##### Dataset Config

Dataset configs are at `./configs/datasets/`.

```yaml
class: datasets.vctk.VCTKDataset # implemented Dataset class name
load:
  audio: "configs/audio/22k.yaml"

path:
  root: data/
  wav22: data/VCTK-Corpus/wav22
  wav16: data/VCTK-Corpus/wav16
  txt: data/VCTK-Corpus/txt

  configs:
    train: data/VCTK-Corpus/vctk_22k_train.txt
    eval: data/VCTK-Corpus/vctk_22k_val.txt
    test: data/VCTK-Corpus/vctk_22k_test.txt
```

#### Model Settings

- Comment out or Delete `Discriminator` section if no Discriminator needed.
- Adjust optimizer `class`, `lr` and `betas` if needed.

```yaml
models:
  Analysis:
    class: models.analysis.Analysis

    optim:
      class: torch.optim.Adam
      kwargs:
        lr: 1e-4
        betas: [0.5, 0.9]

  Synthesis:
    class: models.synthesis.Synthesis

    optim:
      class: torch.optim.Adam
      kwargs:
        lr: 1e-4
        betas: [0.5, 0.9]

  Discriminator:
    class: models.synthesis.Discriminator

    optim:
      class: torch.optim.Adam
      kwargs:
        lr: 1e-4
        betas: [0.5, 0.9]
```

#### Logging & Pytorch-lightning settings

For pytorch-lightning configs in section `pl`, check
[official docs](https://pytorch-lightning.readthedocs.io/en/latest/)

```yaml
pl:
  checkpoint:
    callback:
      save_top_k: -1
      monitor: "train/backward"
      verbose: True
      every_n_epochs: 1 # epochs

  trainer:
    gradient_clip_val: 0 # don't clip (default value)
    max_epochs: 10000
    num_sanity_val_steps: 1
    fast_dev_run: False
    check_val_every_n_epoch: 1
    progress_bar_refresh_rate: 1
    accelerator: "ddp"
    benchmark: True

logging:
  log_dir: logs # PATH TO SAVE TENSORBOARD LOG FILES
  seed: "31" # Experiment Seed
  freq: 100 # Logging frequency (step)
  device: cuda # Training Device (used only in train_torch.py)
  nepochs: 1000 # Max epochs to run

  save_files: # Files To save for each experiment
    [
      "./*.py",
      "./*.sh",
      "configs/*.*",
      "datasets/*.*",
      "models/*.*",
      "utils/*.*",
    ]
```

### Tensorboard

During training, tensorboard logger logs loss, spectrogram and audio.

`tensorboard --logdir YOUR_LOG_DIR_AT_CONFIG/YOUR_SEED --bind_all`
![]('./docs/src/image/tensorboard.png)

## Inference

### Generator

`python inference.py` or `bash inference.sh`

You may want to edit `inferece.py` for custom manipulation.

```python
parser = argparse.ArgumentParser()
parser.add_argument('--path_audio_conf', type=str, default='configs/audio/22k.yaml',
                    help='')
parser.add_argument('--path_ckpt', type=str, required=True,
                    help='path to pl checkpoint')
parser.add_argument('--path_audio_source', type=str, required=True,
                    help='path to source audio file, sr=22k')
parser.add_argument('--path_audio_target', type=str, required=True,
                    help='path to target audio file, sr=16k')
parser.add_argument('--tsa_loop', type=int, default=100,
                    help='iterations for tsa')
parser.add_argument('--device', type=str, default='cuda',
                    help='')
args = parser.parse_args()
return args
```

### Discriminator

Note that 0=gt, 1=gen

`python classify.py` or `bash classify.sh`

```python
parser = argparse.ArgumentParser()
parser.add_argument('--path_audio_conf', type=str, default='configs/audio/22k.yaml',
                    help='')
parser.add_argument('--path_ckpt', type=str, required=True,
                    help='path to pl checkpoint')
parser.add_argument('--path_audio_gt', type=str, required=True,
                    help='path to audio with same speaker')
parser.add_argument('--path_audio_gen', type=str, required=True,
                    help='path to generated audio ')
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()
```

## License

NEEDS WORK

BSD 3-Clause License.

- `model/hifi_gan.py`, `utils/mel.py`, pretrained checkpoints are copied/modified
  from https://github.com/jik876/hifi-gan (MIT License)
- [Wav2Vec2](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#wav2vec-20) (MIT License) pretrained
  checkpoint ported to
  [HuggingFace](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) (Apache License 2.0)

## References

- Choi, Hyeong-Seok, et al. "Neural Analysis and Synthesis: Reconstructing Speech from Self-Supervised Representations."
- Baevski, Alexei, et al. "wav2vec 2.0: A framework for self-supervised learning of speech representations."
- Desplanques, Brecht, Jenthe Thienpondt, and Kris Demuynck. "Ecapa-tdnn: Emphasized channel attention, propagation and
  aggregation in tdnn based speaker verification."
- Chen, Mingjian, et al. "Adaspeech: Adaptive text to speech for custom voice."

- [Cookbook formulae for audio equalizer biquad filter coefficients](https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html)

This implementation uses codes/data from following repositories:

- [hifi-gan](https://github.com/jik876/hifi-gan)
  & [Mellotron](https://github.com/NVIDIA/mellotron/blob/master/yin.py)

Provided Checkpoints are trained from:

- [VCTK Corpus (version 0.92)](https://datashare.ed.ac.uk/handle/10283/3443)
- [LibiTTS train-clean-360](https://openslr.org/60/)
- [CSS10](https://github.com/Kyubyong/css10)

## Special Thanks

[MINDsLab Inc.](https://maum.ai/) for GPU support

Special Thanks to:

- [Junhyeok Lee](https://github.com/junjun3518) @ [MINDsLab Inc.](https://maum.ai/)
- [Seungu Han](https://github.com/Seungwoo0326) @ [MINDsLab Inc.](https://maum.ai/)
- [Kang-wook Kim](https://github.com/wookladin) @ [MINDsLab Inc.](https://maum.ai/)

for help with Audio-domain knowledge
