from ultralytics import YOLO, settings
import sys

# configs
class_conversion = {0: "RBC", 1: "WBC", 2: "Platelets"}

def predict(img_path, model_path="examples/best.pt", show_results=True):
    # Load model
    model = YOLO(model_path)

    # Predict
    results = model.predict(
        img_path, verbose=False
    )

    # get results
    classes_ids = results[0].boxes.cls.cpu().numpy()
    classes_names = [class_conversion[class_id] for class_id in classes_ids]
    positions = results[0].boxes.xywh.cpu().numpy()

    if show_results:
        print("Classes: ", classes_names)
        print("Positions: ", positions)

    return classes_names, positions


if __name__ == "__main__":
    #print(sys.argv[1])
    if len(sys.argv) == 2:
        predict(img_path=sys.argv[1])
    elif len(sys.argv) == 3:
        predict(img_path=sys.argv[1], model_path=sys.argv[2])
    else:
        raise ValueError("Provide the correct number of parameters")

