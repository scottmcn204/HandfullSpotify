import cv2
import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
 

# creating GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Defining Spotify control bash script
bashPlay = "spotify play"
bashPause = "spotify pause"
bashVolUp = "spotify vol up"
bashVolDown = "spotify vol down"
bashPrev = "spotify prev" 
bashNext = "spotify next"

currentGesture = "None"

video = cv2.VideoCapture(0)
while True:
    ret, frame = video.read()
    if not ret:
        break
    cv2.imshow('Video', frame)
    key = key = cv2.waitKey(1)
    if key == 27:
        break
    cv2.imwrite("frame.jpg", frame)
    image = mediapipe.Image.create_from_file("frame.jpg")
    result = recognizer.recognize(image)
    if len(result.gestures) > 0:
        gesture = result.gestures[0][0].category_name
        if(gesture != currentGesture):
            print(gesture)
            if(gesture == "Open_Palm"):
                os.system(bashPause)
            elif(gesture == "Thumb_Up"):
                os.system(bashVolUp)
            elif(gesture == "Thumb_Down"):
                os.system(bashVolDown)
            elif(gesture == "Victory"):
                os.system(bashNext)
            currentGesture = gesture
    # print(result.hand_landmarks)
        
video.release()
cv2.destroyAllWindows()