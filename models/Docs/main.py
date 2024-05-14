from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
import configparser

# Function to compute the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to compute the mouth aspect ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # Upper lip
    B = distance.euclidean(mouth[4], mouth[8])   # Lower lip
    C = distance.euclidean(mouth[0], mouth[6])   # Mouth width
    mar = (A + B) / (2.0 * C)
    return mar

# Function to calculate head pose tilt angle
def head_tilt_angle(landmarks):
    x_diff = landmarks[27][0] - (landmarks[36][0] + landmarks[45][0]) / 2
    if x_diff == 0:
        return 0  # Return 0 angle if the difference is zero to avoid division by zero
    slope = (landmarks[27][1] - (landmarks[36][1] + landmarks[45][1]) / 2) / x_diff
    angle = np.degrees(np.arctan(slope))
    return angle

# Constants for drowsiness detection
EAR_THRESH = 0.24
MAR_THRESH = 0.5
frame_check = 20
angle_max = 90

# Load face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
config = configparser.ConfigParser()
# Video capture
cap = cv2.VideoCapture(0)
flag = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    for rect in rects:
        # Get face coordinates
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        
        # Draw a circle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract eye and mouth landmarks
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        mouth = shape[48:68]

        # Calculate eye aspect ratio (EAR)
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Calculate mouth aspect ratio (MAR)
        mar = mouth_aspect_ratio(mouth)

        # Calculate head tilt angle
        tilt_angle = head_tilt_angle(shape)

        # Draw contours around the eyes and mouth
        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 0, 255), 1)

        # Check for drowsiness
        if ear < EAR_THRESH or mar > MAR_THRESH or abs(tilt_angle) > angle_max:
            flag += 1
            print(flag)
            if flag >= frame_check:
                cv2.putText(frame, "ALERT! Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                config['Detection'] = {
                    'Ear_threshold': str(EAR_THRESH),
                    'Mar_threshold': str(MAR_THRESH),
                    'Drowsiness_detected': 'True'
                }
                with open('drowsiness.ini', 'w') as configfile:
                    config.write(configfile)
        else:
            flag = 0  # Reset the flag to 0 if drowsiness condition is not met

    # Display the frame
    cv2.imshow("MAJOR PROJECT STAGE 2", frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
