# Imports
import face_recognition
import cv2
import numpy as np
import math
from init_face_encodings import known_face_encodings, known_face_names

# Focal-length variables
Known_distances = [1.01, 0.65, 0.33] # New cam
# Known_distances = [0.7, 0.5] # meters # Old cam
Known_width = 0.16 # meters

def FocalLength(measured_distance, real_width, width_in_rf_image):
  focal_length = (width_in_rf_image * measured_distance) / real_width
  return focal_length

# Reading reference images from directory
ref_images = ["focal_length/Ref_image_1010mm_cam2.jpg", "focal_length/Ref_image_650mm_cam2.jpg", "focal_length/Ref_image_330mm_cam2.jpg"] # New cam
focal_lengths = []

# Find focal length from list of reference images with known distances and face width
def calc(face_data):
  for i, ref_image_path in enumerate(ref_images):
      print(f"Loading image: {ref_image_path}")
      ref_image = cv2.imread(ref_image_path)
      if ref_image is None:
          print(f"Error: Unable to read image '{ref_image_path}'")
          continue
      ref_image_face_data = face_data(ref_image, False)
      if ref_image_face_data:
          ref_image_face_width, _, _, _, _ = ref_image_face_data[0]
          focal_length = FocalLength(Known_distances[i], Known_width, ref_image_face_width)
          focal_lengths.append(focal_length)
          print(f"Focal length for {Known_distances[i]} meters: {focal_length}")

  if not focal_lengths:
      print("Error: No valid focal lengths found. Exiting.")
      exit()
      return 0

  # Averaging the focal lengths
  Focal_length_found = np.mean(focal_lengths)
  print(f"Average focal length: {Focal_length_found}")
  return Focal_length_found