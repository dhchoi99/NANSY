# NANSY:

Unofficial Pytorch Implementation of
[Neural Analysis and Synthesis: Reconstructing Speech from Self-Supervised Representations](https://arxiv.org/pdf/2110.14513.pdf)


## ???

### Papers' Demo
Check Authors'
[Demo page](https://harsh-grenadilla-e40.notion.site/Demo-page-for-NANSY-37d4fd8ffb514765a2b234b04c8fc0f6)

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

if using cu113 compatible environment, use [Dockerfile](./Dockerfile)  
if using cu101 compatible environment, use [Dockerfile-cu101](./Dockerfile_cu102)

`docker build -f Dockerfile -t nansy:v0.0 .`

Then,  
`docker-compose -f docker-compose.yml run --service-ports --name CONTAINER_NAME nansy_container bash`  
or  
[`bash docker_run_script.sh`]('./docker_run_script.sh)

### Pretrained hifi-gan

download `config.json`, `generator_v1`
from [hifi-gan](https://github.com/jik876/hifi-gan)
to `./configs/hifi-gan/`

### Pretrained Checkpoints

TODO

## Train

if you prefer `pytorch-lightning`,
`python train.py -g 1`

TODO  
else `python train_torch.py`

## Inference

TODO  
`python inference.py`

## License
