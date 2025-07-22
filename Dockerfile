FROM docker.io/pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel
# FROM docker.io/pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

COPY ./configs ./configs
COPY ./dataloaders/ ./dataloaders
COPY ./flashdepth/ ./flashdepth
COPY ./mamba ./mamba
COPY ./utils/ ./utils

RUN pip install \
    torch==2.4.0 torchvision==0.19.0 \
    xformers==0.0.27.post2 \
    ipdb tqdm wandb \
    matplotlib einops scipy h5py OpenEXR \
    hydra-core \
    opencv-python pillow \
    flash-attn --no-build-isolation && \
    rm -rf ~/.cache/pip

RUN export MAMBA_FORCE_BUILD=TRUE && \
    cd mamba && \
    ls -la && \
    python -m pip install --no-build-isolation . && \
    cd ..

RUN apt-get update && apt-get install -y ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the files that we might need to update
COPY ./train.py .
COPY ./test_entrypoint.sh .

RUN ls -la 
RUN chmod +x ./test_entrypoint.sh

ENTRYPOINT ["torchrun", "train.py"]
# ENTRYPOINT ["bash", "./test_entrypoint.sh"]
# CMD ["--config-path", "configs/flashdepth"]
