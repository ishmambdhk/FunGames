import cv2
import mediapipe as mp
import subprocess

x1 = y1 = x2 = y2 = 0
webcam = cv2.VideoCapture(0)
myHands = mp.solutions.hands.Hands()
drawingUtils = mp.solutions.drawing_utils

def set_volume(volume_level):
    subprocess.run(['osascript', '-e', f'set volume output volume {volume_level}'])

while True:
    _, image = webcam.read()
    frame_height, frame_width, _ = image.shape
    rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = myHands.process(rgbImage)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawingUtils.draw_landmarks(image, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 8:
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 255, 255), thickness=3)
                    x1 = x
                    y1 = y
                if id == 4:
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 0, 255), thickness=3)
                    x2 = x
                    y2 = y

        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (0.5) // 4
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        if dist > 50:
            set_volume(70)
        else:
            set_volume(30)
    cv2.imshow("Hand volume control using Python", image)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()