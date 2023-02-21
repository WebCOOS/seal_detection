import json
from pathlib import Path

import cv2
import requests
import numpy as np
import tensorflow as tf

width = 896
height = 896
threshold = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX

headers = {"content-type": "application/json"}

images = [ str(p) for p in Path(__file__).with_name('inputs').glob("*.jpg") ]

for i in images:

    input_path = Path(i)

    image = cv2.imread(i)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (width , height))
    rgb_tensor = tf.convert_to_tensor(resized, dtype=tf.uint8)
    rgb_tensor = tf.expand_dims(rgb_tensor, 0)
    numpy_data = rgb_tensor.numpy().tolist()

    data = json.dumps({
        "signature_name": "serving_default",
        "instances": numpy_data
    })

    json_response = requests.post(
        'http://localhost:8501/v1/models/seal_detector/versions/3:predict',
        data=data,
        headers=headers
    )

    outputs = json.loads(json_response.text)['predictions']

    for o in outputs:

        output_file = input_path.parent.with_name('outputs') / 'tf_serving' / input_path.name

        boxes = o['output_0']
        boxes = np.asarray(boxes).astype('int')

        scores = o['output_1']
        scores = np.asarray(scores)

        classes = o['output_2']
        classes = np.asarray(classes)

        for score, klass, (ymin,xmin,ymax,xmax) in zip(scores, classes, boxes):

            if score < threshold:
                continue

            h, w, _ = image.shape

            y_min = int(max(1, (ymin * (h / height))))
            x_min = int(max(1, (xmin * (w / width))))
            y_max = int(min(h, (ymax * (h / height))))
            x_max = int(min(w, (xmax * (w / width))))

            img_boxes = cv2.rectangle(
                image,
                (x_min, y_max),
                (x_max, y_min),
                (255, 0, 255, 255),
                thickness=2
            )

            score = round(100 * score, 0)
            label = f"Seal::{score}%"
            cv2.putText(
                img_boxes,
                label,
                (x_min, y_max + 50),
                font,
                1,
                (255, 0, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imwrite(str(output_file), img_boxes)
