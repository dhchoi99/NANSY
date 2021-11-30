## Environment

if using cu113 compatible environment, use [Dockerfile](./Dockerfile)  
if using cu101 compatible environment, use [Dockerfile-cu101](./Dockerfile_cu102)

`docker build -f Dockerfile -t nansy:v0.0 .`

Then,
`docker-compose -f docker-compose.yml run --service-ports --name CONTAINER_NAME dhc_nansy bash`  
or  
[`bash docker_run_script.sh`]('./docker_run_script.sh)

### pretrained hifi-gan

download `config.json`, `generator_v1` 
from [hifi-gan](https://github.com/jik876/hifi-gan) 
to `./configs/hifi-gan/`

## Training

if you prefer `pytorch-lightning`,
`python train.py -g 1`

[comment]: <> (else `python train_torch.py`)