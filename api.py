import os
from typing import Any
from pathlib import Path

import cv2

import requests
import numpy as np
import tensorflow as tf
from pydantic import BaseModel
from fastapi import FastAPI, Depends, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles


app = FastAPI()


class UrlParams(BaseModel):
    url: str


width = 896
height = 896
threshold = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX
headers = {"content-type": "application/json"}

output_path = Path(os.environ.get(
    "OUTPUT_DIRECTORY",
    str(Path(__file__).with_name('outputs') / 'fastapi')
))

model_folder = Path(os.environ.get(
    "MODEL_DIRECTORY",
    str(Path(__file__).parent)
))

models = {
    "seal_detector": {
        "2": tf.saved_model.load(str(model_folder / "seal_detector" / "2")),
        "3": tf.saved_model.load(str(model_folder / "seal_detector" / "3")),
    }
}


def get_model(model: str, version: str):
    return models[model][version]


def process_image(tf_model, model: str, version: str, name: str, bytedata: bytes):
    npdata = np.asarray(bytearray(bytedata), dtype="uint8")
    image = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (width , height))
    rgb_tensor = tf.convert_to_tensor(resized, dtype=tf.uint8)
    rgb_tensor = tf.expand_dims(rgb_tensor, 0)

    # Run the model, get the results
    boxes, scores, classes, num_detections = tf_model(rgb_tensor)

    output_file = output_path / model / str(version) / name

    h, w, _ = image.shape

    boxes = boxes.numpy()[0].astype('int')
    scores = scores.numpy()[0]

    # If any score is above threshold, flag it as detected
    detected = False

    # Draw the results if they are above a defined threshold
    for score, (ymin, xmin, ymax, xmax) in zip(scores, boxes):

        if score < threshold:
            continue

        detected = True

        y_min = int(max(1, (ymin * (h / height))))
        x_min = int(max(1, (xmin * (w / width))))
        y_max = int(min(h, (ymax * (h / height))))
        x_max = int(min(w, (xmax * (w / width))))

        cv2.rectangle(
            image,
            (x_min, y_max),
            (x_max, y_min),
            (255, 0, 255, 255),
            thickness=2
        )

        score = round(100 * score, 0)
        label = f"Seal::{score}%"
        cv2.putText(
            image,
            label,
            (x_min, y_max + 50),
            font,
            1,
            (255, 0, 255, 255),
            2,
            cv2.LINE_AA
        )

    if detected is True:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), image)
        return str(output_file)

    return None

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


@app.post("/{model}/{version}/upload")
def from_upload(
    model: str,
    version: str,
    file: UploadFile,
    tf: Any = Depends(get_model),
):
    bytedata = file.file.read()
    proc = process_image(tf, model, version, file.filename, bytedata)

    rel_path = os.path.relpath( proc, output_path )

    url_for_output = app.url_path_for(
        'outputs', path=rel_path
    )

    return { "url": url_for_output }


@app.post("/{model}/{version}/url")
def from_url(
    model: str,
    version: str,
    params: UrlParams,
    tf: Any = Depends(get_model),
):
    bytedata = requests.get(params.url).content
    name = Path(params.url).name
    proc = process_image(tf, model, version, name, bytedata)
    return { "path": proc }


@app.post("/health")
def health():
    return { "health": "ok" }
