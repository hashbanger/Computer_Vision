#Creating a face smile detector

#importing OpenCV
import cv2

faces_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#Defining the detect function for returning frames after drawing the detector rectangles
def detect(gray, frame):
    #for detecting multiple faces in a frame
    faces = faces_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        
        #Now selecting the face region for smile
        region_gray = gray[y:y+h, x:x+w]
        region_color = frame[y:y+h, x:x+h]
        
        eyes = eyes_cascade.detectMultiScale(region_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(region_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        smile = smile_cascade.detectMultiScale(region_gray, 1.7, 50)
        for (ex, ey, ew, eh) in smile:
            cv2.rectangle(region_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
        
    return frame #Returning the frames with rectangles made

#Now turning on the camera
video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()
        
        