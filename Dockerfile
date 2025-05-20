FROM condaforge/miniforge3

WORKDIR /opt/app
COPY DEEPLABCUT.yaml /opt/app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && conda env update -n base -f DEEPLABCUT.yaml && conda clean -afy

COPY . /opt/app

ENV PATH=/opt/conda/bin:$PATH
RUN chmod -R 777 /opt/app