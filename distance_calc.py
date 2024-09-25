import math

def distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length) / face_width_in_frame
    return distance

def calculate_pixel_distance(f1, f2):
    x1, y1 = f1[3], f1[4]
    x2, y2 = f2[3], f2[4]
    distance_pixels = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance_pixels

def calculate_real_distance(f1_distance, f2_distance, pixel_distance, ref_pixel_width, actual_width):
    avg_face_distance = (f1_distance + f2_distance) / 2
    real_distance = (pixel_distance / ref_pixel_width) * actual_width * avg_face_distance
    return real_distance

def breaking_social_distancing(real_distance):
    return True if real_distance < 2 else False