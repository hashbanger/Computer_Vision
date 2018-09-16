#Basic Face Detection

#importing openCV
import cv2

#Importing the xml files
#We load the cascade for face and eye
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
# We create a function that takes as input the image in black and white (gray) and the 
# original image (frame), and that will return the same image with the detector rectangles. 
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # We apply the detectMultiScale method from the face cascade to locate 
    # one or several faces in the image.
    
    for (x, y, w, h) in faces: #for each detected face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0 ), 2)
        #Here we draw a rectagle of Red color around the face
        
        region_gray = gray[y:y+h, x:x+h] #To select the area of face in gray image
        region_color = frame[y:y+h, x:x+h] #to select the are in coloured images
        #Now going for the eyes
        eyes = eye_cascade.detectMultiScale(region_gray, 1.1, 3)
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(region_color, (ex,ey), (ex+ew, ey+eh ), (0, 255, 0), 2)
    return frame #Returning the coloured frame with rectangles made

#Now Turning On the web cam
video_capture = cv2.VideoCapture(0) # arg 1 for external webcam
             
while True: #Going indefinitely
    _, frame = video_capture.read() #For getting the latest frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converting the frame to gray
    canvas = detect(gray, frame) #getting the output of our detect function
    cv2.imshow('Video', canvas) #Displaying the outputs
    if cv2.waitKey(1) & 0xFF == ord('q'): #To exit the program by pressing q
        break

video_capture.release()  #Safely release the webcam
cv2.destroyAllWindows() #closing the opened windows


       