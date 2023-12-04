from enum import Enum


class ModelFramework(str, Enum):
    TF = "TF"
    YOLO = "YOLO"


class TFModelName(str, Enum):
    seal_detector = "seal_detector"


class TFModelVersion(str, Enum):
    two = "2"
    three = "3"


class YOLOModelName(str, Enum):
    best_seal = "best_seal"


class YOLOModelVersion(str, Enum):
    one = "1"
