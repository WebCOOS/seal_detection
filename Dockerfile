FROM mambaorg/micromamba:1.3.1-jammy-cuda-11.7.1
LABEL MAINTAINER="Kyle Wilcox <kyle@axds.co>"

ENV MODEL_DIRECTORY /models
ENV OUTPUT_DIRECTORY /outputs
ENV APP_DIRECTORY /app

USER root

RUN apt-get update && apt-get install -y \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    mkdir ${APP_DIRECTORY} && \
    chown mambauser:mambauser ${APP_DIRECTORY} && \
    mkdir ${OUTPUT_DIRECTORY} && \
    chown mambauser:mambauser ${OUTPUT_DIRECTORY}

COPY --chown=mambauser:mambauser requirements.txt /tmp/requirements.txt
RUN micromamba install -c conda-forge --name base --yes --file /tmp/requirements.txt && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH "$MAMBA_ROOT_PREFIX/bin:$PATH"

# Copy scripts
COPY --chown=mambauser:mambauser api.py /app/api.py
COPY --chown=mambauser:mambauser seal_detector /models/seal_detector

WORKDIR /app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

