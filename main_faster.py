# Imports
import face_recognition
import cv2
import numpy as np
import math
from init_face_encodings import known_face_encodings, known_face_names
import focal_length_calc, distance_calc, face_calc, display_format

# Camera object, #0 is default/webcam
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output21.mp4", fourcc, 30.0, (640, 480))

# Face detector object
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialise variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

quadrant_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Blue, Green, Red, Yellow

# Focal length variables
Known_width = focal_length_calc.Known_width
Focal_length_found = focal_length_calc.calc(face_calc.face_data)

while True:
    _, frame = cap.read()
    height, width, _ = frame.shape
    faces_data = face_calc.face_data(frame, True)

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

                    pixel_distance = distance_calc.calculate_pixel_distance(face1, face2)
                    ref_pixel_width = face1[0]  # Use the width of the first face as reference
                    actual_width = Known_width  # Real-world width of the face

                    face1_distance = distances.get(face_names[i], 0)
                    face2_distance = distances.get(face_names[j], 0)

                    real_distance = distance_calc.calculate_real_distance(face1_distance, face2_distance, pixel_distance, ref_pixel_width, actual_width)
                    
                    # Draw a line between the two face centers
                    face1_center = (face1[3], face1[4])
                    face2_center = (face2[3], face2[4])

                    # Bounding boxes: (left, top, right, bottom)
                    face1_box = (face1[1], face1[2], face1[1] + face1[0], face1[2] + face1[0])
                    face2_box = (face2[1], face2[2], face2[1] + face2[0], face2[2] + face2[0])

                    # Print names and distance if real distance < 2
                    if distance_calc.breaking_social_distancing(real_distance):
                        print(f"Real distance between {face_names[i]} and {face_names[j]}: {real_distance} meters")
                        frame = display_format.draw_line_with_transparency(frame, face1_center, face2_center, face1_box, face2_box, (255, 0, 0), 2)
                    else:
                        # Draw the line with transparency within the bounding boxes
                        frame = display_format.draw_line_with_transparency(frame, face1_center, face2_center, face1_box, face2_box, (0, 255, 0), 2)

                    # Display the distance on the line
                    midpoint = ((face1_center[0] + face2_center[0]) // 2, (face1_center[1] + face2_center[1]) // 2)
                    w, h = display_format.draw_text(image=frame, text=f"{real_distance:.2f} m", pos=midpoint, font_scale=1, text_color=(255, 255, 255), font_thickness=2)
                    


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
                    Distance = distance_calc.distance_finder(Focal_length_found, Known_width, face_width_in_frame)
                    Distance = round(Distance, 2)
                    distances[name] = Distance
                break

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), quadrant_colors[2], 2)

        # Draw a label with a name below the face
        label = f"{name}, {distances.get(name,'N/A')} m"
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), quadrant_colors[2], cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        box_width = right - left
        box_height = 35  # or any other desired height for the label box
        display_format.put_responsive_text(frame, label, (left, bottom - box_height), box_width, box_height, color=(255, 255, 255))

    # Display image
    cv2.imshow("frame", frame)

    # Quit option, hit 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release handle to webcam
cap.release()
out.release()
cv2.destroyAllWindows()