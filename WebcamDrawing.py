# Importing the required computer vision and numerical array processing modules
import cv2  # Import OpenCV for video capture, image processing, drawing, and UI rendering
import numpy as np  # Import NumPy for array manipulation and defining color bound matrices

# Set Width and Height of output Screen
frameWidth = 640  # Set target frame width to 640 pixels
frameHeight = 480  # Set target frame height to 480 pixels

# Capturing Video from Webcam
cap = cv2.VideoCapture(0)  # Bind to default webcam device at index 0
cap.set(
    3, frameWidth
)  # Set camera frame width property ID 3 (CAP_PROP_FRAME_WIDTH) to 640
cap.set(
    4, frameHeight
)  # Set camera frame height property ID 4 (CAP_PROP_FRAME_HEIGHT) to 480

# Set brightness, ID is 10 and value can be changed accordingly
cap.set(
    10, 150
)  # Set camera brightness property ID 10 (CAP_PROP_BRIGHTNESS) to 150

# Object color values: Define lower and upper HSV color bounds for tracking [Hue_min, Sat_min, Val_min, Hue_max, Sat_max, Val_max]
myColors = [
    [5, 107, 0, 19, 255, 255],  # Orange / Yellow tracking boundary
    [133, 56, 0, 159, 156, 255],  # Purple / Pink tracking boundary
    [57, 76, 0, 100, 255, 255],  # Green tracking boundary
    [90, 48, 0, 118, 255, 255],  # Blue tracking boundary
]

# Color values which will be used to paint on screen (values need to be in BGR format: [Blue, Green, Red])
myColorValues = [
    [51, 153, 255],  # BGR color mapping for Orange/Yellow
    [255, 0, 255],  # BGR color mapping for Purple/Pink
    [0, 255, 0],  # BGR color mapping for Green
    [255, 0, 0],  # BGR color mapping for Blue
]

# Historical point storage array holding elements formatted as: [x_coordinate, y_coordinate, colorId]
myPoints = []


# Function to detect color ranges, create binary masks, and locate pen tips
def findColor(img, myColors, myColorValues):
    imgHSV = cv2.cvtColor(
        img, cv2.COLOR_BGR2HSV
    )  # Convert input frame from BGR to HSV color space
    count = 0  # Initialize color index counter to track matching BGR paint colors
    newPoints = []  # Store newly detected point coordinates for the current frame

    for color in myColors:  # Loop through each target color definition range
        lower = np.array(color[0:3])  # Extract lower HSV bound [H_min, S_min, V_min] into a NumPy array
        upper = np.array(color[3:6])  # Extract upper HSV bound [H_max, S_max, V_max] into a NumPy array
        mask = cv2.inRange(
            imgHSV, lower, upper
        )  # Create binary mask where pixels inside HSV bounds are white (255) and outside are black (0)
        x, y = getContours(
            mask
        )  # Extract top-center coordinate of detected object tip from the mask

        # Draw real-time visual feedback circle on current object position
        cv2.circle(
            imgResult, (x, y), 15, myColorValues[count], cv2.FILLED
        )  # Draw filled circle (radius 15) at tip location

        if (
            x != 0 and y != 0
        ):  # Check if a valid tip location was found (non-zero coordinates)
            newPoints.append(
                [x, y, count]
            )  # Store point coordinates along with color ID
        count += 1  # Increment color index counter for next iteration
    return newPoints  # Return list of detected points in the current frame


# Function to analyze contours, filter noise, and locate the pen tip coordinate
def getContours(img):
    # Check installed OpenCV version to handle API syntax differences for cv2.findContours
    if cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )  # OpenCV 3 return pattern
    else:
        contours, hierarchy = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )  # OpenCV 4+ return pattern

    x, y, w, h = 0, 0, 0, 0  # Initialize bounding box variable values

    for cnt in contours:  # Iterate through all detected contour shapes
        area = cv2.contourArea(cnt)  # Calculate surface pixel area of current contour shape
        if area > 500:  # Noise threshold filter: Ignore small background noise artifacts under 500 pixels
            peri = cv2.arcLength(
                cnt, True
            )  # Calculate total perimeter length of closed contour curve
            approx = cv2.approxPolyDP(
                cnt, 0.02 * peri, True
            )  # Approximate polygonal shape curve with specified epsilon tolerance
            x, y, w, h = cv2.boundingRect(
                approx
            )  # Get axis-aligned bounding rectangle [top-left x, top-left y, width, height]
    return (
        x + w // 2,
        y,
    )  # Return top-center coordinate of bounding box to simulate a pen tip


# Function to redraw all accumulated drawing points onto the screen
def drawOnCanvas(myPoints, myColorValues):
    for point in myPoints:  # Iterate through every stored point entry [x, y, colorId]
        cv2.circle(
            imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED
        )  # Draw filled circle (radius 10)


# Main application loop
while True:
    success, img = cap.read()  # Grab next frame from camera stream
    imgResult = img.copy()  # Create duplicate copy of frame to draw virtual canvas annotations onto

    newPoints = findColor(
        img, myColors, myColorValues
    )  # Search frame for tracked color objects and return new coordinates
    if len(newPoints) != 0:  # Check if new points were detected in current frame
        for newP in newPoints:  # Loop through newly found point entries
            myPoints.append(
                newP
            )  # Append new point to persistent global history list
    if len(myPoints) != 0:  # Check if historical points exist in memory
        drawOnCanvas(
            myPoints, myColorValues
        )  # Render all stored drawing points on canvas

    cv2.imshow("Result", imgResult)  # Display output frame in window titled 'Result'

    # Check for 'q' key press event to exit loop
    if cv2.waitKey(1) and 0xFF == ord('q'):  # Note: Standard syntax is `cv2.waitKey(1) & 0xFF == ord('q')`
        break  # Exit infinite execution loop
