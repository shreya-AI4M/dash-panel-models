from ultralytics import YOLO

model = YOLO("/home/ai4m/shreya/G_block/develop/cropped_holes_dataset_split/runs/detect/runs/yolov9_holes2/weights/best.pt")

results = model.predict(
    source="/home/ai4m/shreya/G_block/develop/cropped_holes_dataset_split/test/images/crop_dash_panel.jpg",
    save=True,
    conf=0.25,
    project="runs",
    name="yolov9_test",
)

for r in results:
    print(f"Detected {len(r.boxes)} objects")
    for box in r.boxes:
        print(f"  Class: {int(box.cls)}, Conf: {box.conf.item():.3f}, Box: {box.xyxy[0].tolist()}")
