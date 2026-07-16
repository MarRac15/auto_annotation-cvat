
"""
yolo_model.py
=============

Train, validate, and export YOLO models for onion/weed detection.

Usage:
    python yolo_model.py --data-path onion_weed_dataset --model yolov8n --epochs 50

This script provides a programmatic interface to Ultralytics YOLO training,
making it easy to integrate into your workflow or call from other scripts.
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.utils import compress_one_image
from ultralytics.utils.downloads import zip_directory


def train_yolo_model(
    data_path: str,
    model_name: str = "yolov8n",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/detect",
    name: str = "train",
    save_period: int = -1,
    val: bool = True,
    plots: bool = True,
):
    """
    Train a YOLO model on the prepared dataset.

    Args:
        data_path: Path to dataset folder (containing data.yaml)
        model_name: YOLO variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        epochs: Number of training epochs
        imgsz: Input image size (pixels)
        batch: Batch size
        project: Project directory for saving results
        name: Experiment name
        save_period: Save checkpoint every N epochs (-1 to disable)
        val: Whether to validate during training
        plots: Whether to save training plots

    Returns:
        Trained YOLO model object
    """
    data_yaml = Path(data_path) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    print(f"Using dataset: {data_yaml}")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}, Image size: {imgsz}, Batch size: {batch}")


    model_path = f"{model_name}.pt"
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Train the model
    print("Starting training...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        save_period=save_period,
        val=val,
        plots=plots,
    )

    print(f"\nTraining complete! Results saved to: {results.save_dir}")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print(f"Last model: {results.save_dir}/weights/last.pt")

    return model


def validate_model(model, data_path: str, split: str = "val"):
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
    results = model.val(data=str(data_yaml), split=split)

    print(f"{split.capitalize()} mAP50: {results.box.map50:.4f}")
    print(f"{split.capitalize()} mAP50-95: {results.box.map:.4f}")
    print(f"{split.capitalize()} Precision: {results.box.mp:.4f}")
    print(f"{split.capitalize()} Recall: {results.box.mr:.4f}")

    return results


def export_model(model, export_format: str = "onnx"):
    """
    Export the trained model to the specified format.

    Args:
        model: Trained YOLO model object
        export_format: Export format (onnx, torchscript, coreml, engine, etc.)

    Returns:
        Path to exported model
    """
    print(f"\n=== Exporting model to {export_format.upper()} ===")
    exported_path = model.export(format=export_format)
    print(f"Exported model saved to: {exported_path}")
    return exported_path


def predict_with_model(model, source: str, conf: float = 0.25, save: bool = True):
    """
    Run inference/prediction with the trained model.

    Args:
        model: Trained YOLO model object
        source: Path to image, video, directory, or URL
        conf: Confidence threshold for detections
        save: Whether to save results

    Returns:
        Prediction results
    """
    print(f"\n=== Running prediction on {source} ===")
    results = model.predict(source=source, conf=conf, save=save)
    print(f"Results saved to: {results[0].save_dir}")
    return results


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_path = os.path.join(project_dir, "runs/detect")
    parser = argparse.ArgumentParser(
        description="Train YOLO model for onion/weed detection (programmatic interface)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        default="yolo_full_dataset",
        help="Path to dataset folder (output of yolo_label.py containing data.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="YOLO model variant",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (pixels)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (adjust for GPU memory)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=runs_path,
        help="Project directory for saving results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Experiment name",
    )
    parser.add_argument(
        "--no-val",
        action="store_false",
        dest="val",
        help="Disable validation during training",
    )
    parser.add_argument(
        "--no-plots",
        action="store_false",
        dest="plots",
        help="Disable saving training plots",
    )
    parser.add_argument(
        "--export",
        type=str,
        default="onnx",
        help="Export format after training (onnx, torchscript, coreml, engine, etc.)",
    )
    parser.add_argument(
        "--validate-test",
        action="store_true",
        help="Run validation on test set after training",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default="",
        help="Run prediction on specified source after training",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions",
    )

    args = parser.parse_args()

    # Train the model
    model = train_yolo_model(
        data_path=args.data_path,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        save_period=-1,
        val=args.val,
        plots=args.plots,
    )

    # Optional: Validate on test set
    if args.validate_test:
        validate_model(model, args.data_path, split="test")

    # Optional: Export model
    if args.export:
        export_model(model, args.export)

    # Optional: Run prediction
    if args.predict:
        predict_with_model(model, args.predict, conf=args.conf)


if __name__ == "__main__":
    main()