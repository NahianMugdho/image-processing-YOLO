import cv2
import numpy as np
from ultralytics import YOLO
import csv
import os
import time
from datetime import datetime

# -----------------------------
# Configurations
# -----------------------------
IMAGE_DIR = r"D:\mugdho\IOT\website\v3Test\client-SIde\client\images"        # Flask কোডের ছবির ফোল্ডার
CSV_FILE = "detection_log.csv"
CONF_THRESHOLD = 0.2
CHECK_INTERVAL = 5           # কত সেকেন্ড পর পর চেক করবে

# YOLO মডেল লোড
model = YOLO("yolov8n.pt")

# যদি CSV না থাকে তাহলে হেডার লেখো
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Image", "Class", "Confidence"])


def process_image(image_path):
    """YOLO দিয়ে একটার ডিটেকশন চালায় এবং রেজাল্ট CSV তে লগ করে।"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image_rgb)[0]

        boxes = results.boxes
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        detections = []
        for box in boxes:
            conf = float(box.conf[0])
            if conf >= CONF_THRESHOLD:
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                detections.append([timestamp, os.path.basename(image_path), class_name, round(conf, 3)])

        if detections:
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(detections)
            print(f"✅ Logged {len(detections)} detections from {os.path.basename(image_path)}")

    except Exception as e:
        print(f"❌ Error processing {image_path}:", e)


def auto_check_folder():
    """নতুন ইমেজ এলে detect করে CSV তে লেখে।"""
    processed = set()

    while True:
        all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
        new_images = [f for f in all_images if f not in processed]

        for img in new_images:
            img_path = os.path.join(IMAGE_DIR, img)
            process_image(img_path)
            processed.add(img)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    print("🔍 Auto detection started... watching 'images/' folder.")
    auto_check_folder()
