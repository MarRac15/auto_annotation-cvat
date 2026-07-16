from ultralytics import YOLO
from pathlib import Path
import os


def validate_model(model, data_path: str, split: str = "val", project="C:/Users/Marek/Documents/runs/detect"):
    """
    Validate the model on a dataset split.

    Args:
        model: Trained YOLO model object
        data_path: Path to dataset folder
        split: Dataset split to validate on ('val' or 'test')

    Returns:
        Validation results object
    """
    data_yaml = Path(data_path) / "data.yaml"
    print(f"\n=== Validating on {split} set ===")
    results = model.val(data=str(data_yaml), split=split, project=project)

    print(f"{split.capitalize()} mAP50: {results.box.map50:.4f}")
    print(f"{split.capitalize()} mAP50-95: {results.box.map:.4f}")
    print(f"{split.capitalize()} Precision: {results.box.mp:.4f}")
    print(f"{split.capitalize()} Recall: {results.box.mr:.4f}")

    return results


def main():

    best_model_path = "C:/Users/Marek/Documents/runs/detect/train/weights/best.pt"
    test_dataset_path = "C:/Users/Marek/Documents/pcss-sophora/yolo_full_dataset"
    model = YOLO(best_model_path)
    results = validate_model(model, test_dataset_path, split="test")


if __name__ == "__main__":
    main()