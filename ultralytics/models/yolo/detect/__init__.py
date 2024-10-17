# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor, DetectionPredictor_2_5 # HWCHU
from .train import DetectionTrainer, DetectionTrainer_2_5 # HWCHU
from .val import DetectionValidator, DetectionValidator_2_5 # HWCHU

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator", "DetectionPredictor_2_5", "DetectionTrainer_2_5", "DetectionValidator_2_5" # HWCHU
