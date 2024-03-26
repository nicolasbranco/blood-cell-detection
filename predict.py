from ultralytics import YOLO, settings
import os

class_convertion = {0: "RBC", 1: "WBC", 2: "Platelets"}

# remove and recreate folders
os.makedirs("yolo_data/datasets", exist_ok=True)
os.makedirs("yolo_data/weights", exist_ok=True)
os.makedirs("yolo_data/runs", exist_ok=True)

# Update a setting
settings.update(
    {
        "datasets_dir": "yolo_data/datasets",
        "weights_dir": "yolo_data/weights",
        "runs_dir": "yolo_data/runs",
    }
)


def predict():
    # Load model
    model = YOLO("yolo_data/runs/detect/train/weights/best.pt")

    # Predict
    results = model.predict(
        "data/processed/Testing/Images/BloodImage_00339.jpg", verbose=False
    )  # or 'data/images', or 'data/images/image.jpg'

    # get results
    classes_ids = results[0].boxes.cls.cpu().numpy()
    classes_names = [class_convertion[class_id] for class_id in classes_ids]
    positions = results[0].boxes.xywh.cpu().numpy()

    # Print results
    print(classes_names)
    print(positions)


if __name__ == "__main__":
    predict()
