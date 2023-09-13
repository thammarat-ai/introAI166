#Example 6. 9 Face Recognition by face_recognition library using Webcam
import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

face0_image = face_recognition.load_image_file("face0.jpeg")
face1_image = face_recognition.load_image_file("face1.jpeg")
face2_image = face_recognition.load_image_file("face2.jpg")
face0_encoding = face_recognition.face_encodings(face0_image)[0]
face1_encoding = face_recognition.face_encodings(face1_image)[0]
face2_encoding = face_recognition.face_encodings(face2_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    face0_encoding,
    face1_encoding,
    face2_encoding
]
known_face_names = [
    "Face 0",
    "Face 1",
    "Face 2"
]

while True:
    ret, frame = video_capture.read()
    # rgb_frame = frame[:, :, ::-1]
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
