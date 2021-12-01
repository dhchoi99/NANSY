FROM nvcr.io/nvidia/pytorch:21.06-py3

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tmux wget sudo git tar htop rsync

RUN python -m pip --no-cache-dir install --upgrade pip setuptools && \
    python -m pip --no-cache-dir install \
    torch==1.10.0+cu113 \
    torchvision==0.11.1+cu113 \
    torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
    python -m pip --no-cache-dir install --upgrade \
    praat-parselmouth \
    transformers==4.12.4 \
    omegaconf \
    pytorch_lightning \
    tqdm \
    librosa \
    tensorboard && \
    python -m pip uninstall -y \
    torchtext tensorboard-plugin-dlprof

COPY . /root/NANSY

WORKDIR /root/NANSY

# cleanup
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /workspace/*
