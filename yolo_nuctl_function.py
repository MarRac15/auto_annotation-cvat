import os
import logging
import base64
import io
import json
from typing import List, Dict, Any
from PIL import Image
from ultralytics import YOLO


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Model loading (runs once per container)
MODEL_PATH = os.getenv("MODEL_PATH", "/models/best.pt")
try:
    _model = YOLO(MODEL_PATH)
    # Extract class names from the model:
    _model_names = _model.names  # {0: 'Onion', 1: 'Weed'}

    logger.info(f"Loaded Ultralytics model from {MODEL_PATH}")
    logger.info(f"Model classes: {_model_names}")

except Exception as exc:
    logger.exception(f"Failed to load model from {MODEL_PATH}: {exc}")
    #create a dummy spec so the function loads (return empty detections)
    _model = None
    _model_names = {}


# Detection thresholds (can be overridden via env)
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
MAX_DET = int(os.getenv("MAX_DET", "300"))

#------------------------------------------
#Nuclio entry point
def handler(context, event):
    """
    Expected JSON payload:
        {"image": "<base64-encoded-png-or-jpg>"}

    Returns JSON:
        {"annotations": [
            {"type": "rectangle", "label_id": <int>, "points": [x1, y1, x2, y2]},
            ...
        ]}
    """
    try:
        if isinstance(event.body, dict):
            data = event.body
        elif isinstance(event.body, (bytes, str)):
            raw = event.body.decode("utf-8") if isinstance(event.body, bytes) else event.body
            data = json.loads(raw)
        else:
            raise ValueError("Event body must be bytes, string, or dict")


        if "image" not in data:
            return _error_response(context, "Missing 'image' field", 400)

        img_b64 = data["image"]
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")


        if _model is None:
            logger.warning("Model not available – returning empty annotation list")
            return _success_response(context, [])

        #returns a list of Results objects (one per image)
        results = _model.predict(
            source=img,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DET,
            verbose=False,
        )

        # -----------------------------------------------------------
        #Convert Results to CVAT annotation list
        annotations: List[Dict[str, Any]] = []
        for result in results:
            # result.boxes holds the detections (xyxy, conf, cls)
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()          # shape (N, 4)
            scores = result.boxes.conf.cpu().numpy()        # shape (N,)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # shape (N,)

            for box, score, cls_id in zip(boxes, scores, class_ids):
                if score < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(float, box)
                label_id = int(cls_id)
                annotations.append(
                    {
                        "confidence": str(score),
                        "label": _model_names[int(cls_id)],
                        "points": [x1, y1, x2, y2],
                        "type": "rectangle",
                    }
                )

        #limit total detections
        if len(annotations) > MAX_DET:
            annotations = sorted(annotations, key=lambda a: a.get("score", 0), reverse=True)[:MAX_DET]

        logger.debug(f"Generated {len(annotations)} annotations")
        return _success_response(context, annotations)

    except Exception as exc:
        logger.exception("Error during detection")
        return _error_response(context, str(exc), 500)


# -----------------------------------------------------------
# Helper builders for responses
def _success_response(context, annotations: List[Dict[str, Any]]):
    import json
    return context.Response(
        body=json.dumps(annotations),
        headers={},
        content_type="application/json",
        status_code=200,
    )


def _error_response(context, message: str, status_code: int):
    import json
    return context.Response(
        body=json.dumps({"error": message}),
        headers={},
        content_type="application/json",
        status_code=status_code,
    )
