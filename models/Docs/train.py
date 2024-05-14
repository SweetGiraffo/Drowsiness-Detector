import cv2
import math
import numpy as np
import dlib
import imutils
from imutils import face_utils
import configparser

def euclideanDist(a, b):
    return math.sqrt(math.pow(a[0]-b[0], 2) + math.pow(a[1]-b[1], 2))

def ear(eye):
    return ((euclideanDist(eye[1], eye[5]) + euclideanDist(eye[2], eye[4])) / (2 * euclideanDist(eye[0], eye[3])))

def mar(mouth):
    return (euclideanDist(mouth[3], mouth[9]) + euclideanDist(mouth[2], mouth[10]) + euclideanDist(mouth[4], mouth[8])) / (3 * euclideanDist(mouth[0], mouth[6]))

def getAvg():
    capture = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    (leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    ear_sum = 0
    mar_sum = 0
    count = 0
    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        if len(rects):
            shape = face_utils.shape_to_np(predictor(gray, rects[0]))
            leftEye = shape[leStart:leEnd]
            rightEye = shape[reStart:reEnd]
            leftEAR = ear(leftEye)
            rightEAR = ear(rightEye)
            ear_sum += (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(gray, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(gray, [rightEyeHull], -1, (0, 255, 0), 1)
            mouth = shape[mStart:mEnd]
            mar_value = mar(mouth)
            mar_sum += mar_value
            cv2.drawContours(gray, [mouth], -1, (0, 255, 0), 1)
            count += 1
        cv2.imshow('Train', gray)
        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
    
    avg_ear = ear_sum / count
    avg_mar = mar_sum / count
    
    return avg_ear, avg_mar

# Call the function to get the average EAR and MAR
avg_ear, avg_mar = getAvg()

# Write the average EAR and MAR values to an INI file
config = configparser.ConfigParser()
config['Averages'] = {'EAR': str(avg_ear), 'MAR': str(avg_mar)}

with open('averages.ini', 'w') as configfile:
    config.write(configfile)
