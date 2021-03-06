docker run \
  -it \
  --ipc=host \
  --gpus=all \
  --network=bridge \
  -p 39960:39960 -p 39961:8888 \
  -v /DATA1/dhchoi:/DATA1/dhchoi \
  -v /home/dhchoi/nansy:/root/nansy \
  --name nansy_container nansy:v0.0-cu102