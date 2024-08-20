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

# Camera object, #0 is default/webcam
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output21.mp4", fourcc, 30.0, (640, 480))

# Face detector object
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length) / face_width_in_frame
    return distance

def face_data(image, CallOut):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    faces_data = []

    for (x, y, w, h) in faces:
        face_center_x = int(w / 2) + x
        face_center_y = int(h / 2) + y

        faces_data.append((w, x, y, face_center_x, face_center_y))
    return faces_data

def calculate_pixel_distance(f1, f2):
    x1, y1 = f1[3], f1[4]
    x2, y2 = f2[3], f2[4]
    distance_pixels = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance_pixels

def calculate_real_distance(f1_distance, f2_distance, pixel_distance, ref_pixel_width, actual_width):
    avg_face_distance = (f1_distance + f2_distance) / 2
    real_distance = (pixel_distance / ref_pixel_width) * actual_width * avg_face_distance
    return real_distance

def put_responsive_text(image, text, position, box_width, box_height, font=cv2.FONT_HERSHEY_DUPLEX, color=(255, 255, 255), thickness=1):
    # Initial font scale and thickness
    font_scale = 1.0
    max_scale_down = 0.1  # How much to reduce the font size if it doesn't fit
    min_font_scale = 0.3  # Minimum font scale to prevent overly small text
    
    # Measure text size with the initial scale
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Adjust the font scale so the text fits the box width
    while text_size[0] > box_width and font_scale > min_font_scale:
        font_scale -= max_scale_down
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Center the text within the bounding box
    text_x = position[0] + (box_width - text_size[0]) // 2
    text_y = position[1] + (box_height + text_size[1]) // 2
    
    # Draw the text
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)

# Reading reference images from directory
ref_images = ["focal_length/Ref_image_1010mm_cam2.jpg", "focal_length/Ref_image_650mm_cam2.jpg", "focal_length/Ref_image_330mm_cam2.jpg"] # New cam
# ref_images = ["focal_length/Ref_image_700mm.jpg", "focal_length/Ref_image_500mm.jpg"] Old cam
focal_lengths = []

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

# Averaging the focal lengths
Focal_length_found = np.mean(focal_lengths)
print(f"Average focal length: {Focal_length_found}")

cv2.imshow("ref_image", ref_image)

# Initialise variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

quadrant_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Blue, Green, Red, Yellow

while True:
    _, frame = cap.read()
    height, width, _ = frame.shape
    faces_data = face_data(frame, True)

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Find all the faces and face encodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        distances = {}

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Ensure there are at least two faces before attempting to calculate distances
    num_faces = len(faces_data)
    if num_faces >= 2:
        for i in range(num_faces):
            for j in range(i + 1, num_faces):
                if i < len(face_names) and j < len(face_names):
                    face1 = faces_data[i]
                    face2 = faces_data[j]

                    pixel_distance = calculate_pixel_distance(face1, face2)
                    ref_pixel_width = face1[0]  # Use the width of the first face as reference
                    actual_width = Known_width  # Real-world width of the face

                    face1_distance = distances.get(face_names[i], 0)
                    face2_distance = distances.get(face_names[j], 0)

                    real_distance = calculate_real_distance(face1_distance, face2_distance, pixel_distance, ref_pixel_width, actual_width)
                    # Draw a line between the two face centers
                    face1_center = (face1[3], face1[4])
                    face2_center = (face2[3], face2[4])
                    cv2.line(frame, face1_center, face2_center, (0, 255, 0), 2)

                    # Display the distance on the line
                    midpoint = ((face1_center[0] + face2_center[0]) // 2, (face1_center[1] + face2_center[1]) // 2)
                    cv2.putText(frame, f"{real_distance:.2f} m", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    print(f"Real distance between {face_names[i]} and {face_names[j]}: {real_distance} meters")

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        face_center_x = (left + right) // 2
        if face_center_x < width // 2:
            color = quadrant_colors[0] if face_center_x < width // 4 else quadrant_colors[1]
        else:
            color = quadrant_colors[2] if face_center_x < 3 * width // 4 else quadrant_colors[3]


        for face_width_in_frame, face_x, face_y, FC_X, FC_Y in faces_data:
            if face_x <= left <= face_x + face_width_in_frame and face_y <= top <= face_y + (bottom - top):
                if face_width_in_frame != 0:
                    Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)
                    Distance = round(Distance, 2)
                    distances[name] = Distance
                break

        # Draw a box around the face"
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        label = f"{name}, {distances.get(name,'N/A')} m"
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        box_width = right - left
        box_height = 35  # or any other desired height for the label box
        put_responsive_text(frame, label, (left, bottom - box_height), box_width, box_height, color=(255, 255, 255))

    # Display image
    cv2.imshow("frame", frame)

    # Quit option, hit 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release handle to webcam
cap.release()
out.release()
cv2.destroyAllWindows()