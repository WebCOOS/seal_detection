
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class BoundingBoxPoint:
    x: float
    y: float


@dataclass
class ClassificationModelResult():

    # Example: YOLO, best_seal, 1
    classification_model_framework: str
    classification_model_name: str
    classification_model_version: str

    detected: bool = False
    detection_count: int = 0

    classification_scores: List[Dict[str, float]] = field(default_factory=list)
    classification_bboxes: List[Tuple[BoundingBoxPoint]] = field(default_factory=list)

    def add(
        self,
        classification_name: str,
        classification_score: float,
        bbox: Tuple[BoundingBoxPoint, BoundingBoxPoint] = None
    ):
        assert classification_name and isinstance( classification_name, str )
        assert isinstance( classification_score, ( float, np.float32, np.float64 ) ), \
            f"classification score should be float, got {classification_score.__class__.__name__}"

        classification_score = float( classification_score )

        if bbox is not None:
            assert isinstance( bbox, tuple )
            assert all( [ isinstance( b, BoundingBoxPoint ) for b in bbox ] )

        self.classification_scores.append(
            {
                classification_name: classification_score
            }
        )

        if bbox is not None:

            self.classification_bboxes.append( bbox )

        self.detection_count += 1

        if( self.detection_count > 0 ):
            self.detected = True

        return
