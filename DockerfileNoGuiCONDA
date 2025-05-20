FROM quay.io/jupyter/base-notebook
COPY requirements.txt /opt/app/requirements.txt
COPY DEEPLABCUT.yaml /opt/app/DEEPLABCUT.yaml

USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
USER $NB_UID:$NB_GID

WORKDIR /opt/app
RUN pip install -r requirements.txt
RUN conda update -n base -c defaults conda
RUN conda env update -n base -f DEEPLABCUT.yaml
#RUN conda env create -f DEEPLABCUT.yaml
COPY . /opt/app

USER root

RUN chown -R ${NB_UID}:${NB_GID} /opt/app
#
#RUN mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/
#RUN echo '{"theme": "JupyterLab Dark"}' > ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings
#
#CMD ["start-notebook.py", \
#     "--NotebookApp.token=''", \
#     "--NotebookApp.notebook_dir=/opt/app/", \
#     "--NotebookApp.default_url=/opt/app/DLC_PIPELINE.ipynb"]