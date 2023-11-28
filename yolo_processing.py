import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging

logger = logging.getLogger( __name__ )



width = 896
height = 896
# threshold = 0.5
# font = cv2.FONT_HERSHEY_SIMPLEX

MODEL_FOLDER = Path(os.environ.get(
    "MODEL_DIRECTORY",
    str(Path(__file__).parent)
))

YOLO_MODELS = {
    "best_seal": {
        "1": YOLO(str(MODEL_FOLDER / "best_seal.pt" )),
    }
}


def yolo_process_image(
    yolo_model,
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

    # npdata = np.asarray(bytearray(bytedata), dtype="uint8")
    # image = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resized = cv2.resize(image, (width , height))
    # rgb_tensor = tf.convert_to_tensor(resized, dtype=tf.uint8)
    # rgb_tensor = tf.expand_dims(rgb_tensor, 0)

    # # Run the model, get the results
    # boxes, scores, _classes, _num_detections = yolo_model(rgb_tensor)

    # output_file = output_path / model / str(version) / name

    # h, w, _ = image.shape

    # boxes = boxes.numpy()[0].astype('int')
    # scores = scores.numpy()[0]

    # # If any score is above threshold, flag it as detected
    # detected = False

    # # Draw the results if they are above a defined threshold
    # for score, (ymin, xmin, ymax, xmax) in zip(scores, boxes):

    #     if score < threshold:
    #         continue

    #     detected = True

    #     y_min = int(max(1, (ymin * (h / height))))
    #     x_min = int(max(1, (xmin * (w / width))))
    #     y_max = int(min(h, (ymax * (h / height))))
    #     x_max = int(min(w, (xmax * (w / width))))

    #     cv2.rectangle(
    #         image,
    #         (x_min, y_max),
    #         (x_max, y_min),
    #         (255, 0, 255, 255),
    #         thickness=2
    #     )

    #     score = round(100 * score, 0)
    #     label = f"Seal::{score}%"
    #     cv2.putText(
    #         image,
    #         label,
    #         (x_min, y_max + 50),
    #         font,
    #         1,
    #         (255, 0, 255, 255),
    #         2,
    #         cv2.LINE_AA
    #     )

    # if detected is True:
    #     output_file.parent.mkdir(parents=True, exist_ok=True)
    #     cv2.imwrite(str(output_file), image)
    #     return str(output_file)

    return None
