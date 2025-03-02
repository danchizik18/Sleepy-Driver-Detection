import cv2
import dlib
import numpy as np
import os
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

EAR_THRESHOLD = 0.2
FRAME_THRESHOLD = 48

frame_count = 0
sleep_alert = False

def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)  
    return ear

results = 'results'
if not os.path.exists(results):
    os.makedirs(results)

file_path = os.path.join(results, f"sleepy_driving_results_{int(time.time())}.txt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        dlib_faces = detector(gray)
        for face in dlib_faces:
            landmarks = predictor(gray, face)

            left_eye = landmarks.parts()[36:42]
            right_eye = landmarks.parts()[42:48]

            left_eye_np = np.array([(point.x, point.y) for point in left_eye])
            right_eye_np = np.array([(point.x, point.y) for point in right_eye])

            all_eye_points = list(left_eye_np) + list(right_eye_np)

            for point in all_eye_points:
                cv2.circle(frame, (point[0], point[1]), 2, (0, 255, 0), -1)

            left_ear = calculate_ear(left_eye_np)
            right_ear = calculate_ear(right_eye_np)

            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                frame_count += 1
            else:
                frame_count = 0  

            if frame_count >= FRAME_THRESHOLD:
                sleep_alert = True
                cv2.putText(frame, "ALERT: Sleepy Driving Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                with open(file_path, 'a') as f:
                    f.write(f"Sleepy Driving Alert Detected at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    cv2.imshow('Driver Sleepiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
