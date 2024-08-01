# Imports
import face_recognition
import cv2
import numpy as np
from init_face_encodings import known_face_encodings, known_face_names

# Focal-length variables
Known_distances = [0.64, 0.65] # meters
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

# Reading reference images from directory
ref_images = ["focal_length/Ref_image_640mm.jpg", "focal_length/Ref_image_650mm.jpg"]
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

while True:
    _, frame = cap.read()
    faces_data = face_data(frame, True)

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Find all the faces and face enqcodings in the frame of video
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

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        for face_width_in_frame, face_x, face_y, FC_X, FC_Y in faces_data:
            if face_x <= left <= face_x + face_width_in_frame and face_y <= top <= face_y + (bottom - top):
                if face_width_in_frame != 0:
                    Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)
                    Distance = round(Distance, 2)
                    distances[name] = Distance
                break

        # Draw a box around the face"
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        label = f"{name}, {distances.get(name,'N/A')} m"
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display image
    cv2.imshow("frame", frame)

    # Quit option, hit 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release handle to webcam
cap.release()
out.release()
cv2.destroyAllWindows()