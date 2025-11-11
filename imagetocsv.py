import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime

image_path = r"D:\mugdho\IOT\website\v3Test\client-SIde\client\images\snapshot_1762829060.jpg"
csv_file = "detection_log.csv"


def detect_objects(image_path):
    """Detect objects in an image using YOLOv8."""
    model = YOLO('yolov8n.pt')
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)[0]

    boxes = results.boxes
    return boxes, results.names, image_rgb


def log_detections_to_csv(boxes, class_names, confidence_threshold, csv_file):
    """Log detections with timestamp into a CSV file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = []

    for box in boxes:
        confidence = float(box.conf[0])
        if confidence > confidence_threshold:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            log_data.append([timestamp, class_name, round(confidence, 3)])

    # Create CSV file if not exists
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Class", "Confidence"])
        writer.writerows(log_data)

    print(f"✅ Logged {len(log_data)} detections to {csv_file}")


def show_results(image_path, confidence_threshold):
    """Show original image and detection results side by side."""
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    boxes, class_names, annotated_image = detect_objects(image_path)

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
    class_labels = {}

    for box in boxes:
        confidence = float(box.conf[0])
        if confidence > confidence_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            color = colors[class_id % len(colors)].tolist()

            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            class_labels[class_name] = color

    # Log detections
    log_detections_to_csv(boxes, class_names, confidence_threshold, csv_file)

    # Visualization
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Detected Objects')
    plt.imshow(annotated_image)
    plt.axis('off')

    legend_handles = []
    for class_name, color in class_labels.items():
        normalized_color = np.array(color) / 255.0
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=class_name,
                                         markerfacecolor=normalized_color, markersize=10))
    plt.legend(handles=legend_handles, loc='upper right', title='Classes')

    plt.tight_layout()
    plt.show()


# Example usage
show_results(image_path, confidence_threshold=0.2)
