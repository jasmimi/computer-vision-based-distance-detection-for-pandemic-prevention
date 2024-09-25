import cv2

# Face detector object
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def face_data(image, CallOut):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    faces_data = []

    for (x, y, w, h) in faces:
        face_center_x = int(w / 2) + x
        face_center_y = int(h / 2) + y

        faces_data.append((w, x, y, face_center_x, face_center_y))
    return faces_data