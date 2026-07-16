
from pathlib import Path

from ultralytics import YOLO


def predict_with_model(model, source: str, project: str, conf: float = 0.25, save: bool = True):
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
    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        project=project
        )
    print(f"Results saved to: {results[0].save_dir}")
    return results



def main():

    best_model_path = "C:/Users/Marek/Documents/runs/detect/train/weights/best.pt"
    model = YOLO(best_model_path)
    model.eval()

    # source = "yolo_full_dataset/images/test"
    # image_extensions = {'.jpg', '.jpeg', '.png'}
    # image_files = [
    #     f for f in Path(source).iterdir() 
    #     if f.is_file() and f.suffix.lower() in image_extensions
    # ]

    # print(f"Found {len(image_files)} images to process")

    # for image_path in image_files:
    #     results = predict_with_model(
    #         model,
    #         source= image_path,
    #         project="C:/Users/Marek/Documents/runs/detect/predict",
    #         conf=0.25
    #         )
    
    test_img = "yolo_full_dataset/images/test/DJI_20260422144322_0243_D_frame_00011.png"
    results = predict_with_model(
            model,
            source= test_img,
            project="C:/Users/Marek/Documents/runs/detect/predict",
            conf=0.25
            )

if __name__ == "__main__":
    main()