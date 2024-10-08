import cv2
import numpy as np

# Variables
Known_distances = [13.976, 26.772, 45.276]  # Inches
Known_width = 5.70866  # Inches

# Colors  >>> BGR Format(BLUE, GREEN, RED)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 242)
GOLDEN = (32, 218, 165)
LIGHT_BLUE = (255, 9, 2)
PURPLE = (128, 0, 128)
CHOCOLATE = (30, 105, 210)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)

fonts = cv2.FONT_HERSHEY_COMPLEX
fonts2 = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
fonts3 = cv2.FONT_HERSHEY_COMPLEX_SMALL
fonts4 = cv2.FONT_HERSHEY_TRIPLEX

# Camera Object
cap = cv2.VideoCapture(0)  # Number According to your Camera

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
        line_thickness = 2
        LLV = int(h * 0.12)
        cv2.line(image, (x, y + LLV), (x + w, y + LLV), (GREEN), line_thickness)
        cv2.line(image, (x, y + h), (x + w, y + h), (GREEN), line_thickness)
        cv2.line(image, (x, y + LLV), (x, y + LLV + LLV), (GREEN), line_thickness)
        cv2.line(image, (x + w, y + LLV), (x + w, y + LLV + LLV), (GREEN), line_thickness)
        cv2.line(image, (x, y + h), (x, y + h - LLV), (GREEN), line_thickness)
        cv2.line(image, (x + w, y + h), (x + w, y + h - LLV), (GREEN), line_thickness)

        face_center_x = int(w / 2) + x
        face_center_y = int(h / 2) + y

        faces_data.append((w, x, y, face_center_x, face_center_y))

        if CallOut:
            cv2.line(image, (x, y - 11), (x + 180, y - 11), (ORANGE), 28)
            cv2.line(image, (x, y - 11), (x + 180, y - 11), (YELLOW), 20)
            cv2.line(image, (x, y - 11), (x + 30, y - 11), (GREEN), 18)

    return faces_data

# Reading reference images from directory
ref_images = ["focal_length/Ref_image_13.jpg", "focal_length/Ref_image_26.jpg", "focal_length/Ref_image_45.jpg"]
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
        print(f"Focal length for {Known_distances[i]} inches: {focal_length}")

if not focal_lengths:
    print("Error: No valid focal lengths found. Exiting.")
    exit()

# Averaging the focal lengths
Focal_length_found = np.mean(focal_lengths)
print(f"Average focal length: {Focal_length_found}")

cv2.imshow("ref_image", ref_image)

while True:
    _, frame = cap.read()
    faces_data = face_data(frame, True)

    for face_width_in_frame, face_x, face_y, FC_X, FC_Y in faces_data:
        if face_width_in_frame != 0:
            Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)
            Distance = Distance * 0.0254  # Convert inches to meters
            Distance = round(Distance, 2)

            # Drawing text on the screen
            cv2.putText(
                frame,
                f"Distance {Distance} Meters",
                (face_x - 6, face_y - 6),
                fonts,
                0.5,
                (BLACK),
                2,
            )

    cv2.imshow("frame", frame)
    out.write(frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
