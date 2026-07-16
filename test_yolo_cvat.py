# test_yolo_cvat.py
from yolo_detect_cvat import detect
from PIL import Image
import cvat_sdk.auto_annotation as cvataa

def test_with_sample_image():
    # Use an actual dataset
    test_image_path = "yolo_full_dataset/images/test/DJI_20260422144322_0243_D_frame_00011.png"

    # Load the image
    image = Image.open(test_image_path)
    print(f"Loaded image: {test_image_path} ({image.size})")

    # Create a mock context (CVAT passes this to your detect function)
    class MockContext:
        def __init__(self):
            self.cancelled = False

    context = MockContext()

    # Run your detection function
    print("\nRunning detection...")
    annotations = detect(context, image)

    # Inspect the first annotation to see what attributes it has
    if annotations:
        first_ann = annotations[0]
        print(f"\nFirst annotation type: {type(first_ann)}")
        print(f"First annotation attributes: {dir(first_ann)}")
        print(f"First annotation __dict__: {first_ann.__dict__ if hasattr(first_ann, '__dict__') else 'No __dict__'}")

        # Try to access common attributes
        attrs_to_try = ['label', 'id', 'points', 'type', 'z_order', 'occluded', 'outside']
        for attr in attrs_to_try:
            if hasattr(first_ann, attr):
                try:
                    value = getattr(first_ann, attr)
                    print(f"  {attr}: {value}")
                except Exception as e:
                    print(f"  {attr}: Error accessing - {e}")
            else:
                print(f"  {attr}: NOT FOUND")

    # Display results
    print(f"\nFound {len(annotations)} annotations:")
    for i, ann in enumerate(annotations):
        # Get label ID from _data_store (CVAT's internal storage)
        label_id = None
        if hasattr(ann, '_data_store'):
            label_id = ann._data_store.get('label_id')
        # Fallback: some CVAT versions may expose label_id as an attribute
        if label_id is None:
            label_id = getattr(ann, 'label_id', None)

        # Get points - try _data_store first, then attribute
        points = None
        if hasattr(ann, '_data_store'):
            points = ann._data_store.get('points')
        if points is None:
            points = getattr(ann, 'points', None)

        # Format label for display
        label_str = str(label_id) if label_id is not None else "UNKNOWN"
        # Format points for display
        if points and len(points) == 4:
            points_str = f"[{', '.join(f'{p:.1f}' for p in points)}]"
        else:
            points_str = "INVALID or missing"

        print(f"  {i+1}. Label ID: {label_str}, Points: {points_str}")

    # Optional: Visualize results
    if annotations:
        # Create a copy to draw on
        from PIL import ImageDraw
        img_with_boxes = image.copy()
        draw = ImageDraw.Draw(img_with_boxes)

        for ann in annotations:
            # Extract points and label ID for drawing
            points = None
            label_id = 0

            if hasattr(ann, '_data_store'):
                points = ann._data_store.get('points')
                label_id = ann._data_store.get('label_id', 0)
            # Fallbacks
            if points is None:
                points = getattr(ann, 'points', None)
            if label_id == 0 and hasattr(ann, 'label_id'):
                label_id = ann.label_id

            if points and len(points) == 4:
                x1, y1, x2, y2 = points
                #draw a rectangle
                #red for onion, green for weed
                color = "red" if label_id == 0 else "green"  
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                # Draw label
                label_text = "Onion" if label_id == 0 else "Weed"
                draw.text((x1, y1-10), label_text, fill=color)

        # Save the result
        output_path = "test_detection_result.jpg"
        img_with_boxes.save(output_path)
        print(f"\nSaved visualization to: {output_path}")
        print("You can open this image to see the detected boxes.")
    else:
        print("\nNo detections found. Try checking your model or confidence threshold.")

if __name__ == "__main__":
    test_with_sample_image()

