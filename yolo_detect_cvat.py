# import site
# site.addsitedir('/home/django/.local/lib/python3.12/site-packages')

import PIL.Image
import cvat_sdk.auto_annotation as cvataa
import os
import logging
from ultralytics import YOLO



logger = logging.getLogger(__name__)


MODEL_PATH = os.environ.get("CVAT_MODEL_PATH", "C:/Users/Marek/Documents/runs/detect/train/weights/best.pt")

# Load model once at startup
try:
    _model = YOLO(MODEL_PATH)
    #_model = ort.InferenceSession(MODEL_PATH)
    logger.info(f"Loaded YOLO model from {MODEL_PATH}")
    logger.info(f"Model classes: {_model.names}")
except Exception as e:
    logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
    raise


spec = cvataa.DetectionFunctionSpec(
    labels=[cvataa.label_spec(name, id) for id, name in _model.names.items()]
)


def _yolo_to_cvat(results):
    """
    Convert YOLOv8 results to CVAT rectangle annotations.

    Args:
        results: YOLO prediction results

    Yields:
        cvataa.rectangle: CVAT rectangle annotations
    """
    for result in results:
        #case where no detections are found
        if result.boxes is None or len(result.boxes) == 0:
            continue

        #extract boxes and class IDs:
        boxes = result.boxes.xyxy  #as pixel coordinates
        classes = result.boxes.cls  #class indices

        # Convert each detection to CVAT format
        for box, cls in zip(boxes, classes):
            # Convert tensor to Python native types
            x1, y1, x2, y2 = [float(coord.item()) for coord in box]
            label_id = int(cls.item())

            yield cvataa.rectangle(label_id, [x1, y1, x2, y2])


def detect(context, image: PIL.Image.Image):
    """
    Main detection function called by CVAT for auto-annotation.

    Args:
        context: CVAT function context (contains configuration parameters)
        image: PIL Image object from CVAT

    Returns:
        List of CVAT annotation objects
    """

    #these can be set via --function-parameter when registering the function
    conf_threshold = getattr(context, 'conf', 0.25)
    iou_threshold = getattr(context, 'iou', 0.45)
    max_det = getattr(context, 'max_det', 300)

    # Log the parameters being used (helpful for debugging)
    logger.debug(f"Detection params - conf: {conf_threshold}, iou: {iou_threshold}, max_det: {max_det}")

    try:
        results = _model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_det,
            verbose=False
        )

        #convert to CVAT format
        annotations = list(_yolo_to_cvat(results))
        logger.debug(f"Generated {len(annotations)} annotations")
        return annotations

    except Exception as e:
        logger.error(f"Error during detection: {e}")
        # Return empty list on error to avoid breaking the annotation process
        return []