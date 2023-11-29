import os
# import requests
from typing import Any
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, Depends, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from tf_processing import tf_process_image, TF_MODELS
from yolo_processing import yolo_process_image, YOLO_MODELS
from namify import namify_for_content
from model_version import (
    TFModelName,
    TFModelVersion,
    YOLOModelName,
    YOLOModelVersion
)
import logging

logger = logging.getLogger( __name__ )


app = FastAPI()


class UrlParams(BaseModel):
    url: str


TF_ENDPOINT_PREFIX = "/tf"
YOLO_ENDPOINT_PREFIX = "/yolo"
ALLOWED_IMAGE_EXTENSIONS = (
    "jpg",
    "png"
)

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
    tags=['tensorflow'],
    summary="Tensorflow/EffDet model prediction on image upload"
)
def tf_from_upload(
    model: TFModelName,
    version: TFModelVersion,
    file: UploadFile,
    tf: Any = Depends(get_tf_model),
):
    """Perform model prediction based on selected Tensorflow model / version."""
    bytedata = file.file.read()

    ( name, ext ) = namify_for_content( bytedata )

    assert ext in ALLOWED_IMAGE_EXTENSIONS, \
        f"{ext} not in allowed image file types: {repr(ALLOWED_IMAGE_EXTENSIONS)}"

    ( res_path, classification_result) = tf_process_image(
        tf,
        output_path,
        model,
        version,
        name,
        bytedata
    )

    if( res_path is None ):
        return {
            "url": None,
            "classification_result": classification_result
        }

    rel_path = os.path.relpath( res_path, output_path )

    url_path_for_output = app.url_path_for(
        'outputs', path=rel_path
    )

    return {
        "url": url_path_for_output,
        "classification_result": classification_result
    }


# @app.post(
#     f"{TF_ENDPOINT_PREFIX}/{{model}}/{{version}}/url",
#     tags=['tensorflow']
# )
# def from_url(
#     model: str,
#     version: str,
#     params: UrlParams,
#     tf: Any = Depends(get_tf_model),
# ):
#     bytedata = requests.get(params.url).content
#     name = Path(params.url).name
#     res = tf_process_image(
#         tf,
#         output_path,
#         model,
#         version,
#         name,
#         bytedata
#     )

#     if( res is None ):
#         return { "url": None }

#     rel_path = os.path.relpath( res, output_path )

#     url_path_for_output = app.url_path_for(
#         'outputs', path=rel_path
#     )

#     return { "url": url_path_for_output }


# YOLO / best_seal.pt endpoints
@app.post(
    f"{YOLO_ENDPOINT_PREFIX}/{{model}}/{{version}}/upload",
    tags=['yolo'],
    summary="Ultralytics/YOLOv8 model prediction on image upload",
)
def yolo_from_upload(
    model: YOLOModelName,
    version: YOLOModelVersion,
    file: UploadFile,
    yolo: Any = Depends(get_yolo_model),
):
    """Perform model prediction based on selected YOLOv8 model / version."""
    bytedata = file.file.read()

    ( name, ext ) = namify_for_content( bytedata )

    assert ext in ALLOWED_IMAGE_EXTENSIONS, \
        f"{ext} not in allowed image file types: {repr(ALLOWED_IMAGE_EXTENSIONS)}"

    ( res_path, classification_result) = yolo_process_image(
        yolo,
        output_path,
        model,
        version,
        name,
        bytedata
    )

    if( res_path is None ):
        return {
            "url": None,
            "classification_result": classification_result
        }

    rel_path = os.path.relpath( res_path, output_path )

    url_path_for_output = app.url_path_for(
        'outputs', path=rel_path
    )

    return {
        "url": url_path_for_output,
        "classification_result": classification_result
    }

# @app.post(
#     f"{YOLO_ENDPOINT_PREFIX}/{{model}}/{{version}}/url",
#     tags=['yolo']
# )
# def from_url(
#     model: str,
#     version: str,
#     params: UrlParams,
#     yolo: Any = Depends(get_yolo_model),
# ):
#     bytedata = requests.get(params.url).content
#     name = Path(params.url).name
#     res = yolo_process_image(
#         yolo,
#         output_path,
#         model,
#         version,
#         name,
#         bytedata
#     )

#     if( res is None ):
#         return { "url": None }

#     rel_path = os.path.relpath( res, output_path )

#     url_path_for_output = app.url_path_for(
#         'outputs', path=rel_path
#     )

#     return { "url": url_path_for_output }


@app.post("/health")
def health():
    return { "health": "ok" }
