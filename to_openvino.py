import os
import shutil
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Load the model
model = YOLO("models/latest_model.pt")

# Export the model to OpenVINO format
export_dir = model.export(format="openvino", half=True)

# Define the destination directory
destination_dir = "models"

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Move the exported files to the destination directory
for file_name in os.listdir(export_dir):
    shutil.move(os.path.join(export_dir, file_name), os.path.join(destination_dir, file_name))
