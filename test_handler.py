"""
Local test for yolo_nuctl_function.py — run this WITHOUT nuclio/Docker
to check your handler logic, model loading, and visually inspect detections
before deploying.

Usage:
    python test_handler.py path/to/test_image.jpg


Set MODEL_PATH env var to point at your local best.pt, e.g.:
    # Windows (PowerShell):
    $env:MODEL_PATH = "C:\\Users\\Marek\\Documents\\runs\\detect\\train\\weights\\best.pt"
    # WSL/Linux:
    export MODEL_PATH=/mnt/c/Users/Marek/Documents/runs/detect/train/weights/best.pt

Output:
    Saves an annotated copy of your image as "<original_name>_annotated.jpg"
    in the same folder, with boxes + label names + confidence drawn on it.
"""

import sys
import os
import base64
import json
import colorsys

if "MODEL_PATH" not in os.environ:
    print("WARNING: MODEL_PATH not set, defaulting to /models/best.pt (likely won't exist locally)")

import yolo_nuctl_function as fn  # noqa: E402
from PIL import Image, ImageDraw, ImageFont


class MockResponse:
    def __init__(self, body, headers, content_type, status_code):
        self.body = body
        self.headers = headers
        self.content_type = content_type
        self.status_code = status_code


class MockContext:
    def Response(self, body, headers, content_type, status_code):
        return MockResponse(body, headers, content_type, status_code)


class MockEvent:
    def __init__(self, body: str):
        self.body = body


def label_color(label_id: int):
    """Deterministic distinct color per label_id."""
    hue = (label_id * 0.61803398875) % 1.0  # golden ratio spacing
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def draw_annotations(image: Image.Image, annotations, class_names: dict) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(14, img.width // 80))
    except Exception:
        font = ImageFont.load_default()

    for ann in annotations:
        x1, y1, x2, y2 = ann["points"]
        label_id = ann["label_id"]
        label_name = class_names.get(label_id, str(label_id))
        color = label_color(label_id)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text = label_name
        try:
            bbox = draw.textbbox((x1, y1), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = font.getsize(text)

        draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, max(0, y1 - th - 4)), text, fill="black", font=font)

    return img


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_handler_visual.py path/to/test_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = json.dumps({"image": img_b64})
    context = MockContext()
    event = MockEvent(body=payload)

    print(f"Model loaded: {fn._model is not None}")
    print(f"Model classes: {fn._model_names}")
    print("Running handler...")

    response = fn.handler(context, event)
    print(f"Status code: {response.status_code}")

    body = json.loads(response.body)
    if "error" in body:
        print(f"Error returned by handler: {body['error']}")
        sys.exit(1)

    annotations = body["annotations"]
    print(f"Found {len(annotations)} detection(s):")
    for ann in annotations:
        name = fn._model_names.get(ann["label_id"], ann["label_id"])
        print(f"  - {name}: {[round(p, 1) for p in ann['points']]}")

    original = Image.open(image_path).convert("RGB")
    annotated = draw_annotations(original, annotations, fn._model_names)

    base, ext = os.path.splitext(image_path)
    out_path = f"{base}_annotated.jpg"
    annotated.save(out_path, quality=95)
    print(f"\nSaved annotated image to: {out_path}")


if __name__ == "__main__":
    main()