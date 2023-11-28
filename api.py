import os
import requests
from typing import Any
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, Depends, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from tf_processing import tf_process_image, TF_MODELS
from yolo_processing import yolo_process_image, YOLO_MODELS


app = FastAPI()


class UrlParams(BaseModel):
    url: str


TF_ENDPOINT_PREFIX = "/tf"
YOLO_ENDPOINT_PREFIX = "/yolo"

output_path = Path(os.environ.get(
    "OUTPUT_DIRECTORY",
    str(Path(__file__).with_name('outputs') / 'fastapi')
))


def get_tf_model(model: str, version: str):
    return TF_MODELS[model][version]


def get_yolo_model(model: str, version: str):
    return YOLO_MODELS[model][version]


# Mounting the 'static' output files for the app
app.mount(
    "/outputs",
    StaticFiles(directory=output_path),
    name="outputs"
)


@app.get("/", include_in_schema=False)
async def index():
    """Convenience redirect to OpenAPI spec UI for service."""
    return RedirectResponse("/docs")


# Tensorflow / EffDet model endpoints
@app.post(
    f"{TF_ENDPOINT_PREFIX}/{{model}}/{{version}}/upload",
    tags=['tensorflow']
)
def from_upload(
    model: str,
    version: str,
    file: UploadFile,
    tf: Any = Depends(get_tf_model),
):
    bytedata = file.file.read()
    res = tf_process_image(
        tf,
        output_path,
        model,
        version,
        file.filename,
        bytedata
    )

    if( res is None ):
        return { "url": None }

    rel_path = os.path.relpath( res, output_path )

    url_path_for_output = app.url_path_for(
        'outputs', path=rel_path
    )

    return { "url": url_path_for_output }


@app.post(
    f"{TF_ENDPOINT_PREFIX}/{{model}}/{{version}}/url",
    tags=['tensorflow']
)
def from_url(
    model: str,
    version: str,
    params: UrlParams,
    tf: Any = Depends(get_tf_model),
):
    bytedata = requests.get(params.url).content
    name = Path(params.url).name
    res = tf_process_image(
        tf,
        output_path,
        model,
        version,
        name,
        bytedata
    )

    if( res is None ):
        return { "url": None }

    rel_path = os.path.relpath( res, output_path )

    url_path_for_output = app.url_path_for(
        'outputs', path=rel_path
    )

    return { "url": url_path_for_output }


# YOLO / best_seal.pt endpoints
@app.post(
    f"{YOLO_ENDPOINT_PREFIX}/{{model}}/{{version}}/upload",
    tags=['yolo']
)
def from_upload(
    model: str,
    version: str,
    file: UploadFile,
    yolo: Any = Depends(get_yolo_model),
):
    bytedata = file.file.read()
    res = yolo_process_image(
        yolo,
        output_path,
        model,
        version,
        file.filename,
        bytedata
    )

    if( res is None ):
        return { "url": None }

    rel_path = os.path.relpath( res, output_path )

    url_path_for_output = app.url_path_for(
        'outputs', path=rel_path
    )

    return { "url": url_path_for_output }


@app.post(
    f"{YOLO_ENDPOINT_PREFIX}/{{model}}/{{version}}/url",
    tags=['yolo']
)
def from_url(
    model: str,
    version: str,
    params: UrlParams,
    yolo: Any = Depends(get_yolo_model),
):
    bytedata = requests.get(params.url).content
    name = Path(params.url).name
    res = yolo_process_image(
        yolo,
        output_path,
        model,
        version,
        name,
        bytedata
    )

    if( res is None ):
        return { "url": None }

    rel_path = os.path.relpath( res, output_path )

    url_path_for_output = app.url_path_for(
        'outputs', path=rel_path
    )

    return { "url": url_path_for_output }


@app.post("/health")
def health():
    return { "health": "ok" }
