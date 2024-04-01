
from prometheus_client import (
    make_asgi_app,
    CollectorRegistry,
    multiprocess,
    Counter
)
from model_version import (
    ModelFramework,
    TFModelName,
    TFModelVersion,
    YOLOModelName,
    YOLOModelVersion
)


OBJECT_CLASSIFICATION_DETECTION_COUNTER = Counter(
    'object_classification_detection_counter',
    'Overall count of inputs with successful detections (that meet a threshold)',
    [
        'model_framework',
        'model_name',
        'model_version',
        'classification_name',
    ]
)

OBJECT_CLASSIFICATION_OBJECT_COUNTER = Counter(
    'object_classification_object_counter',
    'Count of detected objects in all inputs (that meet a threshold)',
    [
        'model_framework',
        'model_name',
        'model_version',
        'classification_name',
    ]
)

# Per: <https://prometheus.github.io/client_python/instrumenting/labels/>
#   Metrics with labels are not initialized when declared, because the client
#   can’t know what values the label can have. It is recommended to initialize
#   the label values by calling the .labels() method alone:
#
#       c.labels('get', '/')

LABELS = (
    ( ModelFramework.TF, TFModelName.seal_detector, TFModelVersion.two ),
    ( ModelFramework.TF, TFModelName.seal_detector, TFModelVersion.three ),
    ( ModelFramework.YOLO, YOLOModelName.best_seal, YOLOModelVersion.one ),
)
__SEAL = 'seal'

for ( fw, mdl, ver ) in LABELS:
    OBJECT_CLASSIFICATION_DETECTION_COUNTER.labels(
        fw.name,
        mdl.value,
        ver.value,
        __SEAL,
    )

    OBJECT_CLASSIFICATION_OBJECT_COUNTER.labels(
        fw.name,
        mdl.value,
        ver.value,
        __SEAL,
    )


def make_metrics_app():
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector( registry )
    return make_asgi_app( registry = registry )


def increment_seal_detection_counter(
    fw: str,
    mdl_name: str,
    mdl_version: str
):
    OBJECT_CLASSIFICATION_DETECTION_COUNTER.labels(
        fw,
        mdl_name,
        mdl_version,
        __SEAL
    ).inc()

def increment_seal_object_counter(
    fw: str,
    mdl_name: str,
    mdl_version: str,
):
    OBJECT_CLASSIFICATION_OBJECT_COUNTER.labels(
        fw,
        mdl_name,
        mdl_version,
        __SEAL
    ).inc()
