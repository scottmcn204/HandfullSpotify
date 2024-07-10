import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2
import numpy
import cv2
import time

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = python.vision.HandLandmarker
HandLandmarkerOptions = python.vision.HandLandmarkerOptions
LandmarkBaseOptions = python.BaseOptions

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def handleGesture(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if len(result.gestures) > 0:
        print('gesture recognition result: {}'.format(result.gestures[0][0].category_name))

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handleGesture)


timeStamp = 0

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

landmarkOptions = HandLandmarkerOptions(
    base_options=LandmarkBaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=2
    )
landmarkDetector = HandLandmarker.create_from_options(landmarkOptions)

def get_annotation_from_frame(frame):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = landmarkDetector.detect(image)
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    return detection_result, annotated_image

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = numpy.copy(rgb_image)
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          mp.solutions.hands.HAND_CONNECTIONS,
          mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
          mp.solutions.drawing_styles.get_default_hand_connections_style())
        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN
        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return annotated_image

with GestureRecognizer.create_from_options(options) as recognizer:
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        detection_result, annotation = get_annotation_from_frame(cv2.flip(frame, 1))
        cv2.imshow('Video', annotation)
        key = key = cv2.waitKey(1)
        if key == 27:
            break
        timeStamp += 1
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognizer.recognize_async(mpImage, timeStamp)

    video.release()
    cv2.destroyAllWindows()
