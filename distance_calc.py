# Imports
import math

# Find face distance from camera
def distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length) / face_width_in_frame
    return distance

# Find distance between faces in pixels
def calculate_pixel_distance(f1, f2):
    x1, y1 = f1[3], f1[4]
    x2, y2 = f2[3], f2[4]
    distance_pixels = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance_pixels

# Find distance between faces in meters
def calculate_real_distance(f1_distance, f2_distance, pixel_distance, ref_pixel_width, actual_width):
    avg_face_distance = (f1_distance + f2_distance) / 2
    real_distance = (pixel_distance / ref_pixel_width) * actual_width * avg_face_distance
    return real_distance

# Set social distancing threshold and see if distance breaches it
def breaking_social_distancing(real_distance):
    SOCIAL_DISTANCE_THRESHOLD = 2
    return True if real_distance < SOCIAL_DISTANCE_THRESHOLD else False