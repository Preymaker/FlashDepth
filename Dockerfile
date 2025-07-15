# FROM python:3.11-slim
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

ENV XDG_RUNTIME_DIR=""

WORKDIR /app

COPY ./configs ./configs
COPY ./dataloaders/ ./dataloaders
COPY ./flashdepth/ ./flashdepth
COPY ./mamba ./mamba
COPY ./utils/ ./utils

RUN pip install torch==2.4.0 torchvision==0.19.0
RUN pip install xformers==0.0.27.post2

RUN pip install ipdb tqdm wandb
RUN pip install matplotlib einops scipy h5py OpenEXR
RUN pip install hydra-core
RUN pip install opencv-python pillow

RUN pip install flash-attn --no-build-isolation
# RUN pip install mamba-ssm --no-build-isolation

RUN export MAMBA_FORCE_BUILD=TRUE && \
    cd mamba && \
    ls -la && \
    python -m pip install --no-build-isolation . && \
    cd ..

RUN apt-get update && apt-get install -y ffmpeg

# Copy the files that we might need to update
COPY ./train.py .
COPY ./test_entrypoint.sh .

RUN ls -la 
RUN chmod +x ./test_entrypoint.sh

ENTRYPOINT ["torchrun", "train.py"]
# ENTRYPOINT ["bash", "./test_entrypoint.sh"]
# CMD ["--config-path", "configs/flashdepth"]
