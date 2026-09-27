import cv2  # Import OpenCV for video capture, image manipulation, and displaying windows
import mediapipe as mp  # Import MediaPipe framework for ML-based hand tracking
from google.protobuf.json_format import MessageToDict  # Import utility to convert C++ Protobuf objects to Python dicts

# Initializing the Model
mpHands = mp.solutions.hands  # Access the Hands module within MediaPipe
hands = mpHands.Hands(  # Instantiate the Hand tracking pipeline with custom parameters
    static_image_mode=False,  # False enables video tracking mode (runs heavy detector once, then tracks frame-to-frame)
    model_complexity=1,  # Sets model capacity (0 = fast/lightweight, 1 = full/balanced accuracy)
    min_detection_confidence=0.75,  # Minimum confidence threshold (75%) required for initial hand detection
    min_tracking_confidence=0.75,  # Minimum confidence threshold (75%) to maintain tracking without re-detecting
    max_num_hands=2)  # Limits model tracking to a maximum of 2 hands at once

# Start capturing video from webcam
cap = cv2.VideoCapture(0)  # Bind to default built-in/USB webcam at device index 0
while True:  # Start continuous frame processing loop
    success, img = cap.read()  # Read next frame (success: bool status, img: NumPy frame array)
    img = cv2.flip(img, 1)  # Flip frame horizontally (along Y-axis) for an intuitive mirror view
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert frame color space from OpenCV BGR to MediaPipe RGB
    results = hands.process(imgRGB)  # Run model inference to extract hand landmarks and handedness metadata

    if results.multi_hand_landmarks:  # Check if at least one hand landmark set was detected
        if len(results.multi_handedness) == 2:  # Check if metadata confirms two distinct hands present
            cv2.putText(img, 'Both Hands', (250, 50),  # Draw 'Both Hands' string at screen coordinate (X: 250, Y: 50)
                        cv2.FONT_HERSHEY_COMPLEX,  # Select standard Hershey Complex font style
                        0.9, (0, 255, 0), 2)  # Font scale: 0.9, Color: Green (BGR: 0, 255, 0), Line thickness: 2px
        else:  # Triggered when only a single hand is detected
            for i in results.multi_handedness:  # Iterate through detected single-hand metadata objects

                # Convert protobuf classification message into dictionary and extract top label ('Left' or 'Right')
                label = MessageToDict(i)['classification'][0]['label']

                if label == 'Left':  # Check if model classifies hand as Left
                    cv2.putText(img, label + ' Hand',  # Overlay 'Left Hand' string on frame
                                (20, 50),  # Place text near top-left area at coordinate (X: 20, Y: 50)
                                cv2.FONT_HERSHEY_COMPLEX,  # Font style
                                0.9, (0, 255, 0), 2)  # Scale: 0.9, Green color, Thickness: 2px

                if label == 'Right':  # Check if model classifies hand as Right
                    cv2.putText(img, label + ' Hand', (460, 50),  # Overlay 'Right Hand' string near top-right (X: 460, Y: 50)
                                cv2.FONT_HERSHEY_COMPLEX,  # Font style
                                0.9, (0, 255, 0), 2)  # Scale: 0.9, Green color, Thickness: 2px

    cv2.imshow('Image', img)  # Render processed frame in a desktop window titled 'Image'
    if cv2.waitKey(1) & 0xff == ord('q'):  # Pause 1ms for keyboard input; check if 8-bit ASCII value equals 'q'
        break  # Exit processing loop if user presses 'q'
