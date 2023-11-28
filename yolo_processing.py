import os
from typing import List, Set
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging

logger = logging.getLogger( __name__ )
width = 896
height = 896
threshold = 0.2
font = cv2.FONT_HERSHEY_SIMPLEX

MODEL_FOLDER = Path(os.environ.get(
    "MODEL_DIRECTORY",
    str(Path(__file__).parent)
))

YOLO_MODELS = {
    "best_seal": {
        "1": YOLO(str(MODEL_FOLDER / "yolo" / "best_seal" / "1" / "best_seal.pt" )),
    }
}


def yolo_process_image(
    yolo_model: YOLO,
    output_path: Path,
    model: str,
    version: str,
    name: str,
    bytedata: bytes
):

    assert yolo_model, \
        f"Must have yolo_model passed to {yolo_process_image.__name__}"

    assert output_path and isinstance( output_path, Path ), \
        f"output_path parameter for {yolo_process_image.__name__} is not Path"

    assert output_path.exists() and output_path.is_dir(), \
        (
            f"output_path parameter for {yolo_process_image.__name__} must exist "
            "and be a directory"
        )

    output_file = output_path / model / str(version) / name

    npdata = np.asarray(bytearray(bytedata), dtype="uint8")
    frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img_boxes = frame

    #use YOLOv8
    results = yolo_model.predict(frame, conf = 0.2)

    # If any score is above threshold, flag it as detected
    detected = False
    scores: List[float] = []
    classes: Set[float] = set()

    for result in results:
        for score, cls, bbox in zip(result.boxes.conf, result.boxes.cls, result.boxes.xyxy):

            scores.append( score.item() )
            classes.add( cls.item() )

            if score < threshold:
                continue

            detected = True

            x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
            h, w, _ = frame.shape

            y_min = int(max(1, y1))
            x_min = int(max(1, x1))
            y_max = int(min(h, y2))
            x_max = int(min(w, x2))

            #label = "Seal" + ": " + ": {:.2f}%".format(score * 100)

            if cls.item() == 0.0:
                label = "Rock" + ": " + ": {:.2f}%".format(score * 100)
                img_boxes = cv2.rectangle(img_boxes, (x_min, y_max), (x_max, y_min), (0, 255, 0), 2)
                cv2.putText(img_boxes, label, (x_min, y_max - 10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            if cls.item() == 1.0:
                label = "Seal" + ": " + ": {:.2f}%".format(score * 100)
                img_boxes = cv2.rectangle(img_boxes, (x_min, y_max), (x_max, y_min), (0, 0, 255), 2)
                cv2.putText(img_boxes, label, (x_min, y_max - 10), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    outp = cv2.resize(img_boxes, (1280, 720))

    logger.warning( f"Scores: {repr( scores )}" )
    logger.warning( f"Classes: {repr( classes )}" )

    if detected is True:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), outp )
        return str(output_file)

    return None
