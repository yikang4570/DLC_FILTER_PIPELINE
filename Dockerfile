FROM jupyter/datascience-notebook:latest

USER root

RUN pip install readyplot

WORKDIR /app

COPY . /app

RUN chown -R 1000:100 /app

RUN fix-permissions /app

EXPOSE 8888

CMD ["start-notebook.sh"]