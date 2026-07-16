import os
import logging
import base64
import io
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
import ast, re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#load model at init (runs once per container)
MODEL_PATH = os.environ.get("MODEL_PATH", "best.onnx")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", "0.45"))
MAX_DET = int(os.environ.get("MAX_DET", "300"))

try:
    # Handle WSL path conversion
    if MODEL_PATH.startswith("/mnt/"):
        windows_path = MODEL_PATH.replace("/mnt/c/", "C:/")
    else:
        windows_path = MODEL_PATH
    
    session = ort.InferenceSession(windows_path)
    
    # Get class names from metadata
    model_meta = session.get_modelmeta()
    custom_metadata = getattr(model_meta, 'custom_metadata_map', {})
    names_str = custom_metadata.get('names', '{0: "object", 1: "object"}')
    
    
    try:
        names_dict = ast.literal_eval(names_str)
    except:
        numbers = re.findall(r'(\d+)\s*:\s*[\'"]([^\'"]+)[\'"]', names_str)
        names_dict = {int(k): v for k, v in numbers} if numbers else {0: 'object', 1: 'object'}
        
    model_names = names_dict
    logger.info(f"Loaded ONNX model from {MODEL_PATH}")
    logger.info(f"Model classes: {model_names}")
    
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    input_shape = session.get_inputs()[0].shape
    input_height = input_shape[2] if len(input_shape) >= 3 else 640
    input_width = input_shape[3] if len(input_shape) >= 4 else 640
    
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

def preprocess_image(image: Image.Image):
    """Preprocess PIL image for YOLOv8 ONNX model."""
    img = image.convert("RGB")
    original_width, original_height = img.size
    
    ratio = min(input_width / original_width, input_height / original_height)
    new_width = int(round(original_width * ratio))
    new_height = int(round(original_height * ratio))
    
    img_resized = img.resize((new_width, new_height))
    img_padded = Image.new("RGB", (input_width, input_height), (114, 114, 114))
    dw = (input_width - new_width) // 2
    dh = (input_height - new_height) // 2
    img_padded.paste(img_resized, (dw, dh))
    
    img_array = np.array(img_padded).transpose(2, 0, 1)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, (original_width, original_height), (dw, dh, ratio)

def postprocess_prediction(outputs, original_size, pad_info):
    """Postprocess YOLOv8 model output."""
    predictions = np.squeeze(outputs[0]).T
    
    boxes = predictions[:, :4]
    scores = np.max(predictions[:, 4:], axis=1) if predictions.shape[1] > 4 else np.ones(len(predictions))
    class_ids = np.argmax(predictions[:, 4:], axis=1) if predictions.shape[1] > 4 else np.zeros(len(predictions), dtype=int)
    
    mask = scores > CONF_THRESHOLD
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return []
    
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    
    dw, dh, ratio = pad_info
    ori_w, ori_h = original_size
    
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= ratio
    
    boxes[:, 0] = np.clip(boxes[:, 0], 0, ori_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, ori_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, ori_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, ori_h)
    
    try:
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            CONF_THRESHOLD,
            IOU_THRESHOLD
        )
    except ImportError:
        indices = []
    
    if len(indices) == 0:
        return []
    
    if isinstance(indices, np.ndarray) and indices.size > 0:
        if indices.ndim > 1:
            indices = indices.flatten()
        boxes = boxes[indices]
        scores = scores[indices]
        class_ids = class_ids[indices]
    
    detections = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        detections.append([x1, y1, x2, y2, float(score), int(class_id)])
    
    return detections

def handler(context, event):
    """Nuclio entry point."""
    try:
        # Expecting JSON with base64 image: {"image": "<base64>"}
        body = event.body
        if isinstance(body, bytes):
            body = body.decode('utf-8')
        
        import json
        data = json.loads(body)
        
        if "image" not in data:
            return context.Response(
                body='{"error": "Missing image field"}',
                headers={},
                content_type='application/json',
                status_code=400
            )
        
        # Decode and process image
        image_bytes = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_bytes))
        
        img_batch, original_size, pad_info = preprocess_image(image)
        outputs = session.run(output_names, {input_name: img_batch})
        detections = postprocess_prediction(outputs, original_size, pad_info)
        
        if len(detections) > MAX_DET:
            detections = sorted(detections, key=lambda x: x[4], reverse=True)[:MAX_DET]
        
        # Format for CVAT
        annotations = []
        for det in detections:
            x1, y1, x2, y2, confidence, class_id = det
            label_name = model_names.get(class_id, str(class_id))
            annotations.append({
                "type": "rectangle",
                "label_id": class_id,
                "label": label_name,
                "points": [x1, y1, x2, y2],
                "group": 0,
                "z_order": 0,
                "occluded": False,
                "outside": False
            })
        
        return context.Response(
            body=json.dumps({"annotations": annotations}),
            headers={},
            content_type='application/json',
            status_code=200
        )
    
    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        return context.Response(
            body=f'{{"error": "{str(e)}"}}',
            headers={},
            content_type='application/json',
            status_code=500
        )