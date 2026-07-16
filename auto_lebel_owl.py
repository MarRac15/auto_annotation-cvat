from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
model     = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16")
model.eval()

IMAGES_PATH = os.path.join(os.getcwd(), "unlabeled")
IMAGE_FILE  = "sample_2.png"
CLASS_NAMES = ["onion", "weed"]

def detect_owlv2(pil_img, thr=0.3):
    texts = [["onion plant or onion leaf"], ["weed plant"]]
    inputs = processor(text=texts, images=pil_img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    target_sizes = torch.Tensor([pil_img.size[::-1]])  # (H, W)
    results = processor.post_process_grounded_object_detection(
        out, target_sizes=target_sizes, threshold=thr
    )[0]
    return results

def show_detections(pil_img: Image, results: dict):
    """
    Display the image with bounding boxes overlay using matplotlib.
    Args:
        pil_img: PIL Image (RGB).
        results: dict from OWL-V2 output containing 'boxes', 'scores', 'labels'.
    """
    # Convert PIL image to numpy array for matplotlib
    img_array = np.array(pil_img)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_array)

    boxes   = results["boxes"]   # tensor [N, 4] (xmin, ymin, xmax, ymax) in absolute pixels
    scores  = results["scores"]  # tensor [N]
    labels  = results["labels"]  # tensor [N] (class indices)


    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        width = x2 - x1
        height = y2 - y1
        cls_id = int(label.item())
        class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"unknown({cls_id})"
        score_val = float(score.item())

        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2,
                                 edgecolor='r', facecolor='none')
        # Add the rectangle to the axes
        ax.add_patch(rect)

        # Add label and score text
        ax.text(x1, y1, f'{class_name}: {score_val:.2f}',
                color='white', fontsize=12,
                bbox=dict(facecolor='red', alpha=0.5, pad=0))

    ax.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    img_path = os.path.join(IMAGES_PATH, IMAGE_FILE)
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = Image.open(img_path).convert("RGB")
    results = detect_owlv2(image, thr=0.15)
    print(f"{len(results['boxes'])} detections:")
    if len(results["boxes"]) == 0:
        print("No objects found above the confidence threshold.")
        return

    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        cls_id = int(label.item())
        class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"unknown({cls_id})"
        print(f"{class_name:12}  score={score.item():.3f}  box={box.tolist()}")

    # Show detections with matplotlib
    show_detections(image, results)

if __name__ == "__main__":
    main()