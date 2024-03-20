FROM mambaorg/micromamba:1.3.1-jammy-cuda-11.7.1
LABEL MAINTAINER="Kyle Wilcox <kyle@axds.co>"

ENV MODEL_DIRECTORY /models
ENV OUTPUT_DIRECTORY /outputs
ENV APP_DIRECTORY /app

ENV PROMETHEUS_MULTIPROC_DIR /tmp/metrics

USER root

RUN apt-get update && apt-get install -y \
        libgl1 \
        libglib2.0-0 \
        libegl-dev \
        libopengl0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    mkdir ${APP_DIRECTORY} && \
    chown mambauser:mambauser ${APP_DIRECTORY} && \
    mkdir ${OUTPUT_DIRECTORY} && \
    chown mambauser:mambauser ${OUTPUT_DIRECTORY} && \
    mkdir ${PROMETHEUS_MULTIPROC_DIR} && \
    chown mambauser:mambauser ${PROMETHEUS_MULTIPROC_DIR}


COPY --chown=mambauser:mambauser environment.yml /tmp/environment.yml
RUN --mount=type=cache,id=webcoos_seal_detector,target=/opt/conda/pkgs \
    --mount=type=cache,id=webcoos_seal_detector,target=/root/.cache/pip \
    micromamba install -c conda-forge --name base --yes --file /tmp/environment.yml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH "$MAMBA_ROOT_PREFIX/bin:$PATH"

# Copy Python app files
COPY --chown=mambauser:mambauser *.py /app/

# Copy container-specific configuration
COPY --chown=mambauser:mambauser docker /docker/
RUN chmod u+x /docker/scripts/expire-annotated-images.sh

# Copy models
COPY --chown=mambauser:mambauser models /models/

WORKDIR /app
CMD ["gunicorn", "api:app", "--config", "/docker/api/gunicorn.conf.py"]
