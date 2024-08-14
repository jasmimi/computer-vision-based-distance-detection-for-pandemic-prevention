# Imports
import face_recognition

# Face encodings
jasmine_image = face_recognition.load_image_file("faces/jasmine.jpg")
jasmine_face_encoding = face_recognition.face_encodings(jasmine_image)[0]

peter_image = face_recognition.load_image_file("faces/peter.jpg")
peter_face_encoding = face_recognition.face_encodings(peter_image)[0]

fucheng_image = face_recognition.load_image_file("faces/fucheng.jpg")
fucheng_face_encoding = face_recognition.face_encodings(fucheng_image)[0]

dylan_image = face_recognition.load_image_file("faces/dylan.jpeg")
dylan_face_encoding = face_recognition.face_encodings(dylan_image)[0]

leo_image = face_recognition.load_image_file("faces/leo.jpeg")
leo_face_encoding = face_recognition.face_encodings(leo_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    jasmine_face_encoding,
    peter_face_encoding,
    fucheng_face_encoding,
    dylan_face_encoding,
    leo_face_encoding
]
known_face_names = [
    "Jasmine Amohia",
    "Peter Chong",
    "Fucheng Zheng",
    "Dylan Logan",
    "Leo Jia"
]