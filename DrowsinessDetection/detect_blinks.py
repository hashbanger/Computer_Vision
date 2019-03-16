from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
    '''Taking the array of eye coordinates and returning the ratio'''
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    
    ear = (A + B)/ (2 * C)
    
    return ear
	
ap = argparse.ArgumentParser()
ap.add_argument('-p','--shape-predictor', required = True,
               help = 'path to the facial landmark predictor')
ap.add_argument('-v','--video', type=str, default="",
               help = 'path to input video file') # omit this for a live video stream
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.30
EYE_AR_CONSEC_FRAMES = 3

#initializing the blink counts and frame counters
COUNTER = 0
TOTAL = 0

print('Loading the facial landmark predictor ###########')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

vs = FileVideoStream(args["video"]).start()
#fileStream = True
vs = VideoStream(src=0).start() # for built in camera
#vs = VideoStream(usePiCamera=True).start()   # for Raspberry pi module
fileStream = False  # use in case of live 
time.sleep(1.0)

# Looping over the frames of the video stream
while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    if fileStream and not vs.more():
        break
        
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width= 450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    for rect in rects: 
        # determining the facial landmarks and then coverting them to a numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        # averaging the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1 )
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1 )
		
        if ear < EYE_AR_THRESH:
            COUNTER += 1 # increasing the blink frame counter (requires EYE_AR_CONSEC_FRAMES to be a blink)

        else:   # the eyes were closed for sufficient frames then consider it as a blink
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0
				
        cv2.putText(frame, "Blinks : {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "E.A.R: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
        
cv2.destroyAllWindows()
vs.stop()